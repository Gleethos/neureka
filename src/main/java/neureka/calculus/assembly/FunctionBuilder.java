package neureka.calculus.assembly;

import neureka.backend.api.Operation;
import neureka.backend.api.operations.OperationContext;
import neureka.calculus.Function;
import neureka.calculus.implementations.FunctionConstant;
import neureka.calculus.implementations.FunctionInput;
import neureka.calculus.implementations.FunctionNode;
import neureka.calculus.implementations.FunctionVariable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class FunctionBuilder
{
    private static final Pattern variablePattern = Pattern.compile("^(-?[iI]{1}[g]?\\[?[ ]*[g]?[jJ]+[ ]*\\]?)");
    private static final Pattern inputPattern    = Pattern.compile("^(-?[iI]{1}[g]?\\[?[ ]*[g]?[0-9]+[ ]*\\]?)");
    private static final Pattern constantPattern = Pattern.compile("^((-{1}[0-9]*|[0-9]*)[.]?[0-9]*(e[-]?[0-9]+)?)");

    private static final Pattern reshapePattern = Pattern.compile("^(\\[{1}(.,)*(.)+[,]?\\]{1}:?((\\({1}[.]*\\){1})|(.+)))");
    private static final Pattern nodePattern = Pattern.compile("^([\\(]{1}.+[\\)]{1})");

    private final OperationContext _context;
    
    public FunctionBuilder(OperationContext context)
    {
        _context = context;
    }
    
    /**
     * @param type
     * @param size
     * @param doAD
     * @return
     */
    public Function build( Operation type, int size, boolean doAD )
    {
        if ( type.isIndexer() ) return build( type.getFunction() + "( I[j] )", doAD );

        String args = IntStream.iterate( 0, n -> n + 1 )
                                .limit( size )
                                .mapToObj( i -> "I[" + i + "]" )
                                .collect( Collectors.joining( ", " ) );

        return build( type.getFunction() + "(" + args + ")", doAD );
    }

    /**
     * @param expression contains the function as String provided by the user
     * @param doAD       is used to turn autograd on or off for this function
     * @return the function which has been built from the expression
     */
    public Function build( String expression, boolean doAD ) {
        expression =
                (expression.length() > 0
                        && (expression.charAt( 0 ) != '(' || expression.charAt( expression.length() - 1 ) != ')'))
                        ? ("(" + expression + ")")
                        : expression;
        String k = ( doAD ) ? "d" + expression : expression;

        if ( Function.CACHE.FUNCTIONS().containsKey( k ) ) return Function.CACHE.FUNCTIONS().get( k );

        expression = FunctionParser.unpackAndCorrect( expression );
        Function built = _build( expression, doAD );
        if ( built != null )
            Function.CACHE.FUNCTIONS().put(
                    ( ( (doAD) ? "d" : "" ) + "(" + built + ")" ).intern(), // Make the String unique!
                    built
            );

        return built;
    }

    /**
     * @param expression is a blueprint String for the function builder
     * @param doAD       enables or disables autograd for this function
     * @return a function which has been built by the given expression
     */
    private Function _build( String expression, boolean doAD )
    {
        expression = expression
                .replace("<<", "" + ((char) 171))
                .replace(">>", "" + ((char) 187));
        expression = expression
                .replace("<-", "<")
                .replace("->", ">");
        Function function;
        ArrayList<Function> sources = new ArrayList<>();
        if ( expression.equals("") ) {
            Function newCore = new FunctionConstant();
            newCore = newCore.newBuild("0");
            return newCore;
        }
        expression = FunctionParser.unpackAndCorrect(expression);
        List<String> foundJunctors = new ArrayList<>();
        List<String> foundComponents = new ArrayList<>();
        int i = 0;
        while ( i < expression.length() ) {
            final String newComponent = FunctionParser.findComponentIn( expression, i );
            if ( newComponent != null ) {
                // Empty strings are not components and will be skipped:
                if ( newComponent.trim().isEmpty()) i += newComponent.length();
                else // String has content so lets add it to the lists:
                {
                    if ( foundComponents.size() <= foundJunctors.size() ) {
                        foundComponents.add(newComponent);
                    }
                    i += newComponent.length(); // And now we continue parsing where the string ends...
                    // After a component however, we expect an operator:
                    final String newOperation = FunctionParser.parsedOperation(expression, i);
                    if ( newOperation != null ) {
                        i += newOperation.length();
                        if ( newOperation.length() <= 0 ) continue;
                        foundJunctors.add( newOperation );
                    }
                }
            }
            else ++i; // Parsing failed for this index so let's try the next one!
        }
        //---
        int counter = _context.id();
        for ( int j = _context.id(); j > 0; --j ) {
            if ( !foundJunctors.contains( _context.instance(j - 1).getOperator() ) )
                --counter;
            else
                j = 0;
        }
        int operationID = 0;
        while ( operationID < counter ) {
            final List<String> newJunctors = new ArrayList<>();
            final List<String> newComponents = new ArrayList<>();
            if ( foundJunctors.contains( _context.instance( operationID ).getOperator() ) ) {
                String currentChain = null;
                boolean groupingOccurred = false;
                boolean enoughPresent = FunctionParser.numberOfOperationsWithin( foundJunctors ) > 1;// Otherwise: I[j]^4 goes nuts!
                if ( enoughPresent ) {
                    String[] ComponentsArray = foundComponents.toArray( new String[ 0 ] );
                    int length = ComponentsArray.length;
                    for ( int ci = 0; ci < length; ci++ ) {
                        String currentComponent;
                        currentComponent = ComponentsArray[ci];
                        String currentOperation = null;
                        if ( foundJunctors.size() > ci ) {
                            currentOperation = foundJunctors.get(ci);
                        }
                        if ( currentOperation != null ) {
                            if ( currentOperation.equals(_context.instance(operationID).getOperator()) ) {
                                final String newChain =
                                        FunctionParser.groupBy(
                                                _context.instance(operationID).getOperator(),
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
                            if ( currentChain == null ) newComponents.add(currentComponent);
                            else {
                                newComponents.add(currentChain + currentComponent);
                                groupingOccurred = true;
                            }
                            currentChain = null;
                        }
                    }
                }
                if (groupingOccurred) {
                    foundJunctors = newJunctors;
                    foundComponents = newComponents;
                }
            }
            operationID += 1;
        }

        // identifying function id:
        int typeId = 0;
        if ( foundJunctors.size() >= 1 ) {
            for ( int id = 0; id < _context.id(); ++id) {
                if ( _context.instance(id).getOperator().equals(foundJunctors.get( 0 )) ) {
                    typeId = id;
                }
            }
        }
        // building sources and function:
        if ( foundComponents.size() == 1 ) {
            String possibleFunction = FunctionParser.parsedOperation(
                    foundComponents.get( 0 ),
                    0
            );
            if ( possibleFunction != null && possibleFunction.length() > 1 ) {

                for ( int oi = 0; oi < _context.id(); oi++ ) {
                    if (_context.instance(oi).getFunction().equalsIgnoreCase(possibleFunction)) {
                        typeId = oi;
                        List<String> parameters = FunctionParser.findParametersIn(
                                                                            foundComponents.get( 0 ),
                                                                            possibleFunction.length()
                                                                    );
                        assert parameters != null;
                        for ( String p : parameters ) sources.add(build(p, doAD));
                        function = new FunctionNode( _context.instance( typeId ), sources, doAD );
                        return function;
                    }
                }
            }
            //---
            String component = FunctionParser.unpackAndCorrect( foundComponents.get( 0 ) );

            if ( constantPattern.matcher( component ).matches() ) return new FunctionConstant().newBuild( component );
            else if ( inputPattern.matcher( component ).find() ) return new FunctionInput().newBuild( component );
            else if ( variablePattern.matcher( component ).find() ) return new FunctionVariable().newBuild( component );
            else if ( component.startsWith("-") ) {
                component = "-1 * "+component.substring(1);
                return _build(component, doAD);
            }

            String cleaned = FunctionParser.cleanedHeadAndTail(component);//If the component did not trigger variable creation: =>Cleaning!
            String raw = component.replace( cleaned, "" );
            String assumed = FunctionParser.assumptionBasedOn( raw );
            if ( assumed.trim().equals("") ) component = cleaned;
            else component = assumed + cleaned;

            return build(component, doAD);
        } else { // More than one component left:
            if ( _context.instance( typeId ).getArity() > 1 ) {
                foundComponents = _groupAccordingToArity(
                                            _context.instance( typeId ).getArity(),
                                            foundComponents,
                                            typeId
                                    );
            } else if (
                    _context.instance(typeId).getOperator().equals(",") &&
                    foundComponents.get( 0 ).startsWith("[")
            ) {
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
            for (String currentComponent2 : foundComponents) {
                Function newCore2 = build(currentComponent2, doAD);//Dangerous recursion lives here!
                sources.add(newCore2);
            }
            sources.trimToSize();
            if (sources.size() == 1) return sources.get( 0 );
            if (sources.size() == 0) return null;
            ArrayList<Function> newVariable = new ArrayList<>();
            for (Function source : sources) {
                if (source != null) newVariable.add(source);
            }
            sources = newVariable;
            function = new FunctionNode(_context.instance(typeId), sources, doAD);
            return function;
        }
    }


    private List<String> _groupAccordingToArity(int arity, List<String> components, int f_id) {
        if ( components.size() > arity && arity > 1 ) {
            String newComponent =
                    "(" +
                            IntStream.iterate(0, n -> n + 1)
                             .limit(arity)
                             .mapToObj( components::get )
                             .collect(Collectors.joining( _context.instance(f_id).getOperator() )) +
                    ")";
            for ( int i = 0; i < arity; i++ )  components.remove(components.get( 0 ));
            components.add(0, newComponent);
            return _groupAccordingToArity(arity, components, f_id);
        }
        return components;
    }

}
