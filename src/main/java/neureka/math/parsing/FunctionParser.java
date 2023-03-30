package neureka.math.parsing;

import neureka.backend.api.BackendContext;
import neureka.backend.api.Operation;
import neureka.math.Function;
import neureka.math.implementations.FunctionConstant;
import neureka.math.implementations.FunctionInput;
import neureka.math.implementations.FunctionNode;
import neureka.math.implementations.FunctionVariable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *  The {@link FunctionParser} takes a {@link BackendContext} instance based on which
 *  it builds {@link Function} implementation instances, usually by parsing {@link String}s.
 *  The information needed for parsing is being provided by the {@link Operation}s within the formerly
 *  mentioned {@link BackendContext}...
 */
public class FunctionParser
{
    private static final Logger _LOG = LoggerFactory.getLogger(FunctionParser.class);

    private static final Pattern _variablePattern = Pattern.compile("^(-?[iI]{1}[g]?\\[?[ ]*[g]?[jJ]+[ ]*\\]?)");
    private static final Pattern _inputPattern = Pattern.compile("^(-?[iI]{1}[g]?\\[?[ ]*[g]?[0-9]+[ ]*\\]?)");
    private static final Pattern _constantPattern = Pattern.compile("^((-?[0-9]*|[0-9]*)[.]?[0-9]*((e|E)[-]?[0-9]+)?)");

    private static final Pattern _permutePattern = Pattern.compile("^(\\[{1}(.,)*(.)+[,]?\\]{1}:?((\\({1}[.]*\\){1})|(.+)))");
    private static final Pattern _nodePattern = Pattern.compile("^([\\(]{1}.+[\\)]{1})");

    private final BackendContext _context;

    /**
     * @param context The {@link BackendContext} which will be used as a basis to parse new {@link Function}
     *                implementation instance from provided {@link String} expressions.
     */
    public FunctionParser( BackendContext context ) { _context = context; }
    
    /**
     * @param operation The {@link Operation} based on which the {@link Function} ought to be created.
     * @param numberOfArgs The number of arguments the produced {@link Function} ought to have.
     * @param doAD The flag determining if the {@link Function} built by this method should perform autograd or not.
     * @return A {@link Function} implementation instance which satisfied the supplied parameters.
     */
    public Function parse( Operation operation, int numberOfArgs, boolean doAD )
    {
        if ( operation.isIndexer() )
            return parse( operation.getIdentifier() + "( I[j] )", doAD );

        String args = IntStream.iterate( 0, n -> n + 1 )
                                .limit( numberOfArgs )
                                .mapToObj( i -> "I[" + i + "]" )
                                .collect( Collectors.joining( ", " ) );

        // A function always has to be parsable:
        return parse( operation.getIdentifier() + "(" + args + ")", doAD );
    }

    /**
     * @param expression contains the function as String provided by the user
     * @param doAD       is used to turn autograd on or off for this function
     * @return the function which has been built from the expression
     */
    public Function parse( String expression, boolean doAD )
    {
        if (
            expression.length() > 0 &&
            (expression.charAt( 0 ) != '(' || expression.charAt( expression.length() - 1 ) != ')')
        )
            expression = ("(" + expression + ")");

        if ( _context.getFunctionCache().has( expression, doAD ) )
            return _context.getFunctionCache().get( expression, doAD );

        expression = ParseUtil.unpackAndCorrect( expression );
        Function built = _parse( expression, doAD );
        if ( built != null )
            _context.getFunctionCache().put( built );
        else
            _LOG.error("Failed to parse function based on expression '"+expression+"' and autograd flag '"+doAD+"'.");
        return built;
    }

    /**
     * @param expression is a blueprint String for the function builder
     * @param doAD       enables or disables autograd for this function
     * @return a function which has been built by the given expression
     */
    private Function _parse( String expression, boolean doAD )
    {
        // TODO: Remove this! It's error prone! (Operations should define parsing to some extent)
        expression = expression
                .replace("<<", "" + ((char) 171))
                .replace(">>", "" + ((char) 187));
        expression = expression
                .replace("<-", "<")
                .replace("->", ">");

        if ( expression.equals("") )
            return new FunctionConstant("0");

        expression = ParseUtil.unpackAndCorrect(expression);
        List<String> foundOperations = new ArrayList<>();
        List<String> foundComponents = new ArrayList<>();

        for ( int ei = 0; ei < expression.length(); ) {
            final String newComponent = ParseUtil.findComponentIn( expression, ei );
            if ( newComponent != null ) {
                // Empty strings are not components and will be skipped:
                if ( newComponent.trim().isEmpty()) ei += newComponent.length();
                else // String has content so lets add it to the lists:
                {
                    if ( foundComponents.size() <= foundOperations.size() ) {
                        foundComponents.add(newComponent);
                    }
                    ei += newComponent.length(); // And now we continue parsing where the string ends...
                    // After a component however, we expect an operator:
                    final String newOperation = ParseUtil.parsedOperation( expression, ei );
                    if ( newOperation != null ) {
                        ei += newOperation.length();
                        if ( newOperation.length() <= 0 ) continue;
                        foundOperations.add( newOperation );
                    }
                }
            }
            else
                ++ei; // Parsing failed for this index so let's try the next one!
        }
        //---

        int counter = _context.size();
        for ( int j = _context.size(); j > 0; --j ) {
            if ( !foundOperations.contains( _context.getOperation(j - 1).getOperator() ) )
                --counter;
            else
                j = 0;
        }
        for ( int operationID = 0; operationID < counter; operationID++ ) {
            final List<String> newJunctors = new ArrayList<>();
            final List<String> newComponents = new ArrayList<>();
            if ( foundOperations.contains( _context.getOperation( operationID ).getOperator() ) ) {
                String currentChain = null;
                boolean groupingOccurred = false;
                boolean enoughPresent = ParseUtil.numberOfOperationsWithin( foundOperations ) > 1;// Otherwise: I[j]**4 goes nuts!
                if ( enoughPresent ) {
                    String[] foundCompArray = foundComponents.toArray( new String[ 0 ] );
                    int length = foundCompArray.length;
                    for ( int ci = 0; ci < length; ci++ ) {
                        String currentComponent;
                        currentComponent = foundCompArray[ci];
                        String currentOperation = null;
                        if ( foundOperations.size() > ci ) {
                            currentOperation = foundOperations.get(ci);
                        }
                        if ( currentOperation != null ) {
                            if (
                                    currentOperation.equals(_context.getOperation(operationID).getOperator())
                            ) {
                                final String newChain =
                                        ParseUtil.groupBy(
                                                _context.getOperation(operationID).getOperator(),
                                                currentChain,
                                                currentComponent,
                                                currentOperation
                                        );
                                if ( newChain != null ) currentChain = newChain;

                                groupingOccurred = true;
                            } else {
                                if ( currentChain == null ) newComponents.add(currentComponent);
                                else newComponents.add( currentChain + currentComponent );
                                newJunctors.add( currentOperation );
                                groupingOccurred = true;
                                currentChain = null;
                            }
                        } else {
                            if ( currentChain == null )
                                newComponents.add( currentComponent );
                            else {
                                newComponents.add( currentChain + currentComponent );
                                groupingOccurred = true;
                            }
                            currentChain = null;
                        }
                    }
                }
                if ( groupingOccurred ) {
                    foundOperations = newJunctors;
                    foundComponents = newComponents;
                }
            }
        }

        // building sources and function:
        if ( foundComponents.size() == 1 )
            return _buildFunction( foundComponents.get(0), doAD );
        else
            // It's not a function but operators:
            return _buildOperators( foundComponents, foundOperations, doAD );

    }

    private Function _buildFunction( String foundComponent, boolean doAD ) {
        ArrayList<Function> sources = new ArrayList<>();
        String possibleFunction = ParseUtil.parsedOperation(
                foundComponent,
                0
        );
        if ( possibleFunction != null && possibleFunction.length() > 1 ) {
            for ( int oi = 0; oi < _context.size(); oi++ ) {
                if (_context.getOperation(oi).getIdentifier().equalsIgnoreCase(possibleFunction)) {
                    List<String> parameters = ParseUtil.findParametersIn(
                                                                        foundComponent,
                                                                        possibleFunction.length()
                                                                );
                    assert parameters != null;
                    for ( String p : parameters ) sources.add(parse(p, doAD));
                    return new FunctionNode( _context.getOperation( oi ), sources, doAD );
                }
            }
        }
        //---
        String component = ParseUtil.unpackAndCorrect( foundComponent );

        if ( _constantPattern.matcher( component ).matches()   ) return new FunctionConstant( component );
        else if ( _inputPattern.matcher( component ).find()    ) return FunctionInput.of( component );
        else if ( _variablePattern.matcher( component ).find() ) return new FunctionVariable( component );
        else if ( component.startsWith("-") ) {
            component = "-1 * "+component.substring(1);
            return _parse(component, doAD);
        }
        // If the component did not trigger constant/input/variable creation: -> Cleaning!
        String cleaned = ParseUtil.cleanedHeadAndTail( component );
        String raw = component.replace( cleaned, "" );
        String assumed = ParseUtil.assumptionBasedOn( raw );
        if ( assumed.trim().equals("") ) component = cleaned;
        else component = assumed + cleaned;
        // Let's try again:
        Function result;
        try {
            result = parse( component, doAD );
        } catch (Exception e) {
            throw new IllegalStateException("Failed to parse expression '"+component+"'! Cause: "+e.getCause());
        }
        return result;
    }

    private Function _buildOperators(
            List<String> foundComponents,
            List<String> foundOperators,
            boolean doAD
    ) {
        // identifying operator id:
        int operationIndex = 0;
        if ( foundOperators.size() >= 1 ) {
            for (int currentIndex = 0; currentIndex < _context.size(); ++currentIndex) {
                if ( _context.getOperation(currentIndex).getOperator().equals(foundOperators.get( 0 )) ) {
                    operationIndex = currentIndex;
                }
            }
        }
        String asString = foundComponents.stream()
                .collect(
                        Collectors.joining(
                                _context.getOperation( operationIndex ).getOperator()
                        )
                );
        // More than one component left:
        ArrayList<Function> sources = new ArrayList<>();
        if ( _context.getOperation( operationIndex ).getArity() > 1 ) {
            foundComponents = _groupAccordingToArity(
                                        _context.getOperation( operationIndex ).getArity(),
                                        foundComponents,
                                        _context.getOperation( operationIndex ).getOperator()
                                );
        } else if ( _permutePattern.matcher(asString).matches() ) {
            foundComponents.set(0, foundComponents.get( 0 ).substring(1));
            String[] splitted;
            if (foundComponents.get(foundComponents.size() - 1).contains("]")) {
                int offset = 1;
                if (foundComponents.get(foundComponents.size() - 1).contains("]:")) {
                    offset = 2;
                    splitted = foundComponents.get(foundComponents.size() - 1).split("]:");
                } else {
                    splitted = foundComponents.get(foundComponents.size() - 1).split("]");
                }
                if (splitted.length > 1) {
                    splitted = new String[]{splitted[ 0 ], foundComponents.get(foundComponents.size() - 1).substring(splitted[ 0 ].length() + offset)};
                    foundComponents.remove(foundComponents.size() - 1);
                    foundComponents.addAll(Arrays.asList(splitted));
                }
            }
        }
        for ( String component : foundComponents )
            sources.add(
                    parse(component, doAD) // a dangerous recursion lives here!
            );

        sources.trimToSize();
        if ( sources.size() == 1 ) return sources.get( 0 );
        if ( sources.size() == 0 ) return null;
        ArrayList<Function> newVariable = new ArrayList<>();
        for ( Function source : sources ) {
            if ( source != null ) newVariable.add(source);
        }
        sources = newVariable;
        return new FunctionNode( _context.getOperation(operationIndex), sources, doAD );
    }

    private List<String> _groupAccordingToArity(int arity, List<String> components, String operator) {
        if ( components.size() > arity && arity > 1 ) {
            String newComponent =
                    "(" +
                            IntStream.iterate( 0, n -> n + 1 )
                             .limit(arity)
                             .mapToObj( components::get )
                             .collect(Collectors.joining( operator )) +
                    ")";
            for ( int i = 0; i < arity; i++ )  components.remove(components.get( 0 ));
            components.add(0, newComponent);
            return _groupAccordingToArity( arity, components, operator );
        }
        return components;
    }

}
