package neureka.calculus.assembly;

import neureka.backend.api.Operation;
import neureka.backend.api.operations.OperationContext;
import org.jetbrains.annotations.Contract;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Utility for parsing function expressions.
 **/
public class FunctionParser
{
    @Contract( pure = true ) 
    public static int numberOfOperationsWithin( final List<String> operations ) {
        int counter = 0;
        for( Operation ot : OperationContext.get().instances() ) {
            if (operations.contains(ot.getOperator())) ++counter;
        }
        return counter;
    }

    @Contract( pure = true )
    public static String parsedOperation( final String exp, final int index ) {
        if (exp.length() <= index) return null;
        String operation = "";
        for ( int i = exp.length()-1; i >= index; i--) {
            operation = exp.substring(index, i);
            if (FunctionParser.isAnOperation(operation) || FunctionParser.isAnOperation(operation.toLowerCase())) {
                return operation;
            }
        }
        return null;
    }

    @Contract( pure = true )
    public static String findComponentIn( String exp, final int index ) {
        exp = exp.trim();
        if (exp.length() <= index) return null;
        int bracketDepth = 0;
        StringBuilder component = new StringBuilder();
        for ( int i = index; i < exp.length(); ++i)
        {
            if ( exp.charAt( i ) == ')' ) --bracketDepth;
            else if ( exp.charAt( i ) == '(' ) ++bracketDepth;
            if ( bracketDepth == 0 ) {
                String possibleOperation;
                for ( int ii = exp.length()-1; ii >= i+1; ii--) {
                    String found = FunctionParser.parsedOperation( exp.substring( i, ii ), i );
                    if (
                         found != null && // If the found string is a function then we continue!
                                 !OperationContext.get().instance(found).getOperator().equals(found)
                    ) {
                        ii = -1; // end inner loop
                        component.append( found, 0, found.length() - 1 );
                        i += found.length()-1;
                    } else {
                        possibleOperation = exp.substring( i + 1, ii );
                        if ( FunctionParser.isAnOperation( possibleOperation ) ) {
                            if (
                                    ( exp.charAt( i ) == 'j' || !Character.isLetter( exp.charAt( i ) ) )
                            ) {
                                component.append( exp.charAt( i ) );
                                return component.toString();
                            }
                        }
                    }
                }
            }
            component.append(exp.charAt( i ));
        }
        return component.toString();
    }

    @Contract( pure = true )
    public static List<String> findParametersIn( String exp, final int index ) {
        exp = exp.trim();
        if (exp.length() <= index) return null;
        int bracketDepth = 0;
        List<String> parameters = new ArrayList<>();
        StringBuilder component = new StringBuilder();
        for ( int i = index; i < exp.length(); ++i)
        {
            if ( exp.charAt( i ) == '(' || exp.charAt( i ) == '[' ) {
                if ( bracketDepth != 0 ) component.append(exp.charAt( i ));
                ++bracketDepth;
            } else if ( exp.charAt( i ) == ')' || exp.charAt( i ) == ']' ) {
                --bracketDepth;
                if ( bracketDepth != 0 ) component.append(exp.charAt( i ));
            } else if ( exp.charAt( i ) != ',' || bracketDepth > 1 ) { // Use depth!
                component.append( exp.charAt( i ) );
            }
            if ( bracketDepth == 0 ) {
                parameters.add( component.toString() );
            } else if ( bracketDepth == 1 && exp.charAt( i ) == ',' ) {
                parameters.add( component.toString() );
                component = new StringBuilder();
            }
        }
        return parameters;
    }

    @Contract( pure = true )
    public static boolean isAnOperation( final String operationName ) {
        if ( operationName.length() > 32 ) return false;
        Operation operation = OperationContext.get().instance( operationName );
        return operation != null && operation.getId() >= 0;
    }

    @Contract( pure = true )
    public static String groupBy(
            final String operation,
            final String currentChain,
            final String currentComponent,
            final String currentOperation
    ) {
        String group = null;
        if (currentOperation != null) {
            if (currentOperation.equals(operation)) {
                group = currentComponent + currentOperation;
                if (currentChain != null) group = currentChain + group;
            }
        } else if (currentChain != null) group = currentChain + currentComponent;
        return group;
    }

    @Contract( pure = true )
    private static boolean isForbiddenChar( char c ) {
        return c == '"' || c == '$' || c == '%' || c == '&' || c == '=' || c == '#' || c == '|' || c == '~' || c == ':'
                || c == ';' || c == '@' || c == '?' || c == '\\' || c == '>' || c == '<' || c == ' ';
    }

    @Contract( pure = true )
    public static String cleanedHeadAndTail( String exp ) {
        exp = exp.trim();
        int ci = 0;
        StringBuilder updated = new StringBuilder();
        boolean condition = true;
        while ( condition ) {
            if (FunctionParser.isForbiddenChar(exp.charAt(ci)) || (exp.charAt(ci) >= 'A' && exp.charAt(ci) <= 'Z') || (exp.charAt(ci) >= 'a' && exp.charAt(ci) <= 'z')) {
                ci++;
            } else condition = false;
            if (ci == exp.length()) condition = false;
        }
        for ( int gi = ci; gi < exp.length(); gi++) updated.append(exp.charAt(gi));
        exp = updated.toString();
        updated = new StringBuilder();
        if (exp.length() > 0) {
            ci = 0;
            condition = true;
            int l = exp.length() - 1;
            while ( condition ) {
                if (
                        isForbiddenChar( exp.charAt( ci ) ) ||
                        ( exp.charAt( l - ci ) >= 'A' && exp.charAt( l - ci ) <= 'Z' ) ||
                        ( exp.charAt( l - ci ) >= 'a' && exp.charAt( l - ci ) <= 'z' )
                ) {
                    ci++;
                } else condition = false;
                if ( l - ci < 0 ) condition = false;
            }
            for ( int gi = 0; gi <= l - ci; gi++) updated.append( exp.charAt(gi) );
            exp = updated.toString();
        }
        if ( exp.length() > 0 ) {
            if ( exp.charAt( 0 ) == '(' && exp.charAt( exp.length() - 1 ) != ')' ) {
                exp = exp.substring(1, exp.length()-1);
            }
            if ( exp.charAt(exp.length() - 1) == ')' && exp.charAt( 0 ) != '(' ) {
                exp = exp.substring(1, exp.length()-1);
            }
        }
        exp = exp.trim();
        return exp;
    }

    @Contract( pure = true )
    public static String unpackAndCorrect( String exp ) {
        if ( exp == null ) return null;
        if ( exp.length() == 0 ) return "";
        if ( exp.equals("()") ) return "";
        exp = exp.trim();
        exp = exp.replace("sigmoid", "sig");
        exp = exp.replace("quadratic", "quad");
        exp = exp.replace("quadr", "quad");
        exp = exp.replace("lig", "softplus");
        exp = exp.replace("ligmoid", "softplus");
        exp = exp.replace("splus", "softplus");
        exp = exp.replace("spls", "softplus");
        exp = exp.replace("ligm", "softplusd");
        exp = exp.replace("identity", "idy");
        exp = exp.replace("ident", "idy");
        exp = exp.replace("self", "idy");
        exp = exp.replace("copy", "idy");
        exp = exp.replace("gaussian", "gaus");
        exp = exp.replace("gauss", "gaus");
        exp = exp.replace("absolute", "abs");
        exp = exp.replace("summation", "sum");
        exp = exp.replace("product", "prod");

        int bracketDepth = 0;
        for ( int Ei = 0; Ei < exp.length(); ++Ei) {
            if (exp.charAt(Ei) == '(') ++bracketDepth;
            else if (exp.charAt(Ei) == ')') --bracketDepth;
        }
        if (bracketDepth != 0) {
            if (bracketDepth < 0) {
                StringBuilder expBuilder = new StringBuilder(exp);
                for ( int Bi = 0; Bi < -bracketDepth; ++Bi) {
                    expBuilder.insert(0, "(");
                }
                exp = expBuilder.toString();
            }
            else
                exp = new StringBuilder(exp).append(
                        String.join("", Collections.nCopies( bracketDepth, ")" )) // repeat!
                ).toString();
        }
        boolean parsing = true;
        boolean needsStitching = false;
        while (parsing && (exp.charAt( 0 ) == '(') && (exp.charAt(exp.length() - 1) == ')')) {
            bracketDepth = 0;
            needsStitching = true;
            for ( int i = 0; i < exp.length(); ++i) {
                if (exp.charAt( i ) == ')') --bracketDepth;
                else if (exp.charAt( i ) == '(') ++bracketDepth;
                if (bracketDepth == 0 && i != exp.length() - 1) needsStitching = false;
            }
            if (needsStitching) exp = exp.substring(1, exp.length()-1);
            else parsing = false;
        }
        return exp.trim();
    }

    /**
     *  This method tries to find the next best operation {@link String} the user might have meant.
     *
     * @param expression
     * @return
     */
    @Contract( pure = true )
    public static String assumptionBasedOn( String expression ) {
        double largest = -1;
        int best = 0;
        for ( int i = 0; i< OperationContext.get().id(); i++ ) {
            double s = similarity( expression, OperationContext.get().instance( i ).getOperator() );
            if ( largest == -1 ) largest = s;
            else if (s > largest) {
                best = i;
                largest = s;
            }
        }
        return ( largest > 0.1 ) ? OperationContext.get().instance(best).getOperator() : "";
    }

    /**
     *  This method estimates the similarity between 2 provided {@link String} instances.
     *
     * @param s1
     * @param s2
     * @return
     */
    @Contract( pure = true )
    public static double similarity( final String s1, final String s2 ) {
            String longer = (s1.length() > s2.length()) ?s1 : s2;
            String shorter = (s1.length() > s2.length()) ? s2 : s1;
            // longer should always have greater length
            if ( longer.length() == 0 ) return 1.0; /* both strings are zero length */

            int delta = (longer.length()-shorter.length());
            double[] alignment = new double[ delta + 1 ];
            double[] weights = new double[ delta + 1 ];
            double currentWeight = longer.length();
            double weightSum = 0;
            double modifier = delta / (double) longer.length();
            for ( int i = 0; i < ( delta + 1 ); i++ ) {
                weights[ i ] = currentWeight;
                weightSum += currentWeight;
                currentWeight *= modifier;
                for ( int si = 0; si < shorter.length(); si++ ) {
                    char lChar = longer.charAt( i + si );
                    char sChar = shorter.charAt( si );
                    if ( lChar == sChar ) alignment[ i ] ++;
                    else if ( // Custom modifiers:
                        Character.toLowerCase( lChar ) == Character.toLowerCase( sChar )
                    ) alignment[ i ] += 0.5;
                    else if (
                        Character.isAlphabetic( lChar ) != Character.isAlphabetic( sChar )
                    ) alignment[ i ] -= 0.13571113;
                }
                alignment[ i ] /= longer.length();
                alignment[ i ] = Math.min( Math.max( alignment[ i ], 0.0 ), 1.0 );
            }
            Arrays.sort( alignment );
            Arrays.sort( weights );
            double similarity = 0;
            for ( int i = 0; i < ( delta + 1 ); i++ ) similarity += alignment[ i ] * ( weights[ i ] / weightSum );
            assert similarity <= 1.0;
            return similarity;
    }


}
