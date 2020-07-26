package neureka.calculus.factory.assembly;

import neureka.calculus.environment.OperationType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Utility for parsing
 * */
public class FunctionParser
{
    public static int numberOfOperationsWithin(final List<String> operations) {
        int counter = 0;
        for(OperationType ot : OperationType.instances()){
            if (operations.contains(ot.identifier())) ++counter;
        }
        return counter;
    }

    public static String parsedOperation(final String exp, final int index) {
        if (exp.length() <= index) return null;
        String operation = "";
        for (int i = exp.length()-1; i >= index; i--) {
            operation = exp.substring(index, i);
            if (FunctionParser.isBasicOperation(operation)) {
                return operation;
            }
        }
        return null;
    }

    public static String findComponentIn(String exp, final int index) {
        exp = exp.trim();
        if (exp.length() <= index) return null;
        int bracketDepth = 0;
        StringBuilder component = new StringBuilder();
        for (int i = index; i < exp.length(); ++i)
        {
            if (exp.charAt(i) == ')') --bracketDepth;
            else if (exp.charAt(i) == '(') ++bracketDepth;
            if (bracketDepth == 0) {
                String possibleOperation = "";
                for (int ii = exp.length()-1; ii >= i+1; ii--) {
                    possibleOperation = exp.substring(i+1, ii);
                    if (FunctionParser.isBasicOperation(possibleOperation)) {
                        if (exp.charAt(i)=='j' || !Character.isLetter(exp.charAt(i))) {
                            component.append(exp.charAt(i));
                            return component.toString();
                        }
                    }
                }
            }
            component.append(exp.charAt(i));
        }
        return component.toString();
    }

    public static List<String> findParametersIn(String exp, final int index) {
        exp = exp.trim();
        if (exp.length() <= index) return null;
        int bracketDepth = 0;
        List<String> parameters = new ArrayList<>();
        StringBuilder component = new StringBuilder();
        for (int i = index; i < exp.length(); ++i)
        {
            if (exp.charAt(i) == '(' || exp.charAt(i) == '[') {
                if (bracketDepth != 0) component.append(exp.charAt(i));
                ++bracketDepth;
            } else if (exp.charAt(i) == ')' || exp.charAt(i) == ']') {
                --bracketDepth;
                if (bracketDepth != 0) component.append(exp.charAt(i));
            } else {
                component.append(exp.charAt(i));
            }
            if (bracketDepth == 0) {
                parameters.add(component.toString());
            } else if (bracketDepth == 1 && exp.charAt(i)==',' ) {
                parameters.add(component.toString());
                component = new StringBuilder();
            }
        }
        return parameters;
    }

    public static boolean isBasicOperation(final String operation) {
        if (operation.length() > 8) return false;
        return (OperationType.instance(operation) != null) && OperationType.instance(operation).id() >= 0;
    }

    public static String groupBy(final String operation, final String currentChain, final String currentComponent, final String currentOperation) {
        String group = null;
        if (currentOperation != null) {
            if (currentOperation.equals(operation)) {
                group = currentComponent + currentOperation;
                if (currentChain != null) group = currentChain + group;
            }
        } else if (currentChain != null) group = currentChain + currentComponent;
        return group;
    }

    private static boolean isForbiddenChar(char c) {
        return c == '"' || c == '$' || c == '%' || c == '&' || c == '=' || c == '#' || c == '|' || c == '~' || c == ':'
                || c == ';' || c == '@' || c == '?' || c == '\\' || c == '>' || c == '<' || c == ' ';
    }

    public static String cleanedHeadAndTail(String exp) {
        exp = exp.trim();
        int ci = 0;
        StringBuilder updated = new StringBuilder();
        boolean condition = true;
        while (condition) {
            if (FunctionParser.isForbiddenChar(exp.charAt(ci)) || (exp.charAt(ci) >= 'A' && exp.charAt(ci) <= 'Z') || (exp.charAt(ci) >= 'a' && exp.charAt(ci) <= 'z')) {
                ci++;
            } else condition = false;
            if (ci == exp.length()) condition = false;
        }
        for (int gi = ci; gi < exp.length(); gi++) updated.append(exp.charAt(gi));
        exp = updated.toString();
        updated = new StringBuilder();
        if (exp.length() > 0) {
            ci = 0;
            condition = true;
            int l = exp.length() - 1;
            while (condition) {
                if (FunctionParser.isForbiddenChar(exp.charAt(ci)) || (exp.charAt(l - ci) >= 'A' && exp.charAt(l - ci) <= 'Z') || (exp.charAt(l - ci) >= 'a' && exp.charAt(l - ci) <= 'z')) {
                    ci++;
                } else condition = false;
                if (l - ci < 0) condition = false;
            }
            for (int gi = 0; gi <= l - ci; gi++) updated.append(exp.charAt(gi));
            exp = updated.toString();
        }
        if (exp.length() > 0) {
            if (exp.charAt(0) == '(' && exp.charAt(exp.length() - 1) != ')') {
                exp = exp.substring(1, exp.length()-1);
            }
            if (exp.charAt(exp.length() - 1) == ')' && exp.charAt(0) != '(') {
                exp = exp.substring(1, exp.length()-1);
            }
        }
        exp = exp.trim();
        return exp;
    }

    public static String unpackAndCorrect(String exp) {
        if ( exp == null ) return null;
        if ( exp.length() == 0 ) return "";
        if ( exp.equals("()") ) return "";
        exp = exp.trim();
        exp = exp.replace("sigmoid", "sig");//Function.TYPES.REGISTER(1));
        exp = exp.replace("quadratic", "quad");//Function.TYPES.REGISTER(3));
        exp = exp.replace("quadr", "quad");//Function.TYPES.REGISTER(3));
        exp = exp.replace("lig", "lig");//Function.TYPES.REGISTER(4));
        exp = exp.replace("ligmoid", "lig");//Function.TYPES.REGISTER(4));
        exp = exp.replace("softplus", "lig");//Function.TYPES.REGISTER(4));
        exp = exp.replace("spls", "lig");//Function.TYPES.REGISTER(4));
        exp = exp.replace("ligm", "lig");//Function.TYPES.REGISTER(4));
        exp = exp.replace("identity", "idy");//Function.TYPES.REGISTER(5));
        exp = exp.replace("ident", "idy");//Function.TYPES.REGISTER(5));
        exp = exp.replace("self", "idy");//Function.TYPES.REGISTER(5));
        exp = exp.replace("copy", "idy");//Function.TYPES.REGISTER(5));
        exp = exp.replace("gaussian", "gaus");//Function.TYPES.REGISTER(6));
        exp = exp.replace("gauss", "gaus");//Function.TYPES.REGISTER(6));
        exp = exp.replace("absolute", "abs");//Function.TYPES.REGISTER(7));
        exp = exp.replace("summation", "sum");//Function.TYPES.REGISTER(10));
        exp = exp.replace("product", "prod");//Function.TYPES.REGISTER(11));

        int bracketDepth = 0;
        for (int Ei = 0; Ei < exp.length(); ++Ei) {
            if (exp.charAt(Ei) == '(') ++bracketDepth;
            else if (exp.charAt(Ei) == ')') --bracketDepth;
        }
        if (bracketDepth != 0) {
            if (bracketDepth < 0) {
                StringBuilder expBuilder = new StringBuilder(exp);
                for (int Bi = 0; Bi < -bracketDepth; ++Bi) {
                    expBuilder.insert(0, "(");
                }
                exp = expBuilder.toString();
            } else exp = new StringBuilder(exp).append(")".repeat(bracketDepth)).toString();
        }
        boolean parsing = true;
        boolean needsStitching = false;
        while (parsing && (exp.charAt(0) == '(') && (exp.charAt(exp.length() - 1) == ')')) {
            bracketDepth = 0;
            needsStitching = true;
            for (int i = 0; i < exp.length(); ++i) {
                if (exp.charAt(i) == ')') --bracketDepth;
                else if (exp.charAt(i) == '(') ++bracketDepth;
                if (bracketDepth == 0 && i != exp.length() - 1) needsStitching = false;
            }
            if (needsStitching) exp = exp.substring(1, exp.length()-1);
            else parsing = false;
        }
        return exp.trim();
    }

    public static String assumptionBasedOn(String expression){
        double largest = -1;
        int best = 0;
        for (int i=0; i<OperationType.COUNT(); i++){
            double s = similarity(expression, OperationType.instance(i).identifier());
            if (largest==-1) largest = s;
            else if (s > largest) {
                best = i;
                largest = s;
            }
        }
        return ( largest > 0.1 ) ? OperationType.instance(best).identifier() : "";
    }

    public static double similarity(String s1, String s2) {
            String longer = (s1.length() > s2.length()) ?s1 : s2;
            String shorter = (s1.length() > s2.length()) ? s2 : s1;
            // longer should always have greater length
            if (longer.length() == 0) return 1.0; /* both strings are zero length */

            int delta = (longer.length()-shorter.length());
            double[] alignment = new double[delta+1];
            double[] weights = new double[delta+1];
            double currentWeight = longer.length();
            double weightSum = 0;
            double modifier = delta / (double)longer.length();
            for ( int i=0; i<(delta+1); i++ ){
                weights[i] = currentWeight;
                weightSum += currentWeight;
                currentWeight *= modifier;
                for (int si=0; si<shorter.length(); si++) {
                    if (longer.charAt(i+si)==shorter.charAt(si)) alignment[i] ++;
                    else if (
                            Character.toLowerCase(longer.charAt(i+si)) == Character.toLowerCase(shorter.charAt(si))
                    ) alignment[i] += 0.5;
                    else if (
                            Character.isAlphabetic(longer.charAt(i+si)) != Character.isAlphabetic(shorter.charAt(si))
                    ) alignment[i] -= 0.13571113;
                }
                alignment[i] /= longer.length();
                alignment[i] = Math.min(Math.max(alignment[i], 0.0), 1.0);
            }
            Arrays.sort(alignment);
            Arrays.sort(weights);
            double similarity = 0;
            for (int i=0; i<(delta+1); i++) similarity += alignment[i] * (weights[i]/weightSum);
            System.out.println(longer+" ~?~ "+shorter+" : "+similarity);
            assert similarity <= 1.0;

            return similarity;

    }


}
