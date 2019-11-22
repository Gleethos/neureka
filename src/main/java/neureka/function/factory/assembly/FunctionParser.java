package neureka.function.factory.assembly;

import neureka.function.Function;

import java.util.List;
import java.util.ListIterator;

/**
 * Utility for parsing
 * */
public class FunctionParser {
    public static int numberOfOperationsWithin(final List<String> operations) {
        int Count = 0;
        for (int i = 0; i < Function.TYPES.REGISTER.length; ++i) {
            if (FunctionParser.containsOperation(Function.TYPES.REGISTER[i], operations)) {
                ++Count;
            }
        }
        return Count;
    }

    public static String parsedOperation(final String exp, final int i) {
        if (exp.length() <= i) {
            return null;
        }
        String operation = "";
        for (int Si = i; Si < exp.length(); ++Si) {
            operation = (operation) + exp.charAt(Si);
            if (FunctionParser.isBasicOperation(operation)) {
                return operation;
            }
        }
        return null;
    }

    public static String parsedComponent(String exp, final int i) {
        exp = exp.trim();
        if (exp.length() <= i) {
            return null;
        }
        String component = "";
        int bracketDepth = 0;
        component = "";
        for (int Ei = i; Ei < exp.length(); ++Ei) {
            if (exp.charAt(Ei) == ')') {
                --bracketDepth;
            } else if (exp.charAt(Ei) == '(') {
                ++bracketDepth;
            }
            if (bracketDepth == 0) {
                String possibleOperation = "";
                for (int Sii = Ei+1; Sii < exp.length(); ++Sii) {
                    possibleOperation = possibleOperation + exp.charAt(Sii);
                    if (FunctionParser.isBasicOperation(possibleOperation)) {
                        component = component + exp.charAt(Ei);
                        return component;
                    }
                }
            }
            component += exp.charAt(Ei);
        }
        return component;
    }

    public static boolean containsOperation(final String operation, final List<String> operations) {
        final ListIterator<String> OperationIterator = operations.listIterator();
        while (OperationIterator.hasNext()) {
            final String currentOperation = OperationIterator.next();
            if (currentOperation.equals(operation)) {
                return true;
            }
        }
        return false;
    }

    public static boolean isBasicOperation(final String operation) {
        if (operation.length() > 8) {
            return false;
        }
        for (int i = 0; i < Function.TYPES.REGISTER.length; ++i) {
            if (Function.TYPES.REGISTER[i].equals(operation)) {
                return true;
            }
        }
        return false;
    }

    public static String groupBy(final String operation, final String currentChain, final String currentComponent, final String currentOperation) {
        String group = null;
        if (currentOperation != null) {
            if (currentOperation.equals(operation)) {
                group = currentComponent + currentOperation;
                if (currentChain != null) {
                    group = currentChain + group;
                }
            }
        } else if (currentChain != null) {
            group = currentChain + currentComponent;
        }
        return group;
    }

    private static boolean isWeired(char c) {
        if (c == '"' || c == '$' || c == '%' || c == '&' || c == '=' || c == '#' || c == '|' || c == '~' || c == ':'
                || c == ';' || c == '@' || c == '?' || c == '\\' || c == '>' || c == '<' || c == ' ') {
            return true;
        }
        return false;
    }

    public static String removeHeadAndTail(final String exp) {
        String corrected = "";
        for (int i = 1; i < exp.length() - 1; ++i) {
            corrected = corrected + exp.charAt(i);
        }
        return corrected;
    }

    public static String cleanedHeadAndTail(String exp) {
        exp = exp.trim();
        int Ci = 0;
        String Updated = "";
        boolean condition = true;
        while (condition) {
            if (FunctionParser.isWeired(exp.charAt(Ci)) || (exp.charAt(Ci) >= 'A' && exp.charAt(Ci) <= 'Z') || (exp.charAt(Ci) >= 'a' && exp.charAt(Ci) <= 'z')) {
                Ci++;
            } else {
                condition = false;
            }
            if (Ci == exp.length()) {
                condition = false;
            }
        }
        for (int Gi = Ci; Gi < exp.length(); Gi++) {
            Updated += exp.charAt(Gi);
        }
        exp = Updated;
        Updated = "";
        if (exp.length() > 0) {
            Ci = 0;
            condition = true;
            int l = exp.length() - 1;
            while (condition) {
                if (FunctionParser.isWeired(exp.charAt(Ci)) || (exp.charAt(l - Ci) >= 'A' && exp.charAt(l - Ci) <= 'Z') || (exp.charAt(l - Ci) >= 'a' && exp.charAt(l - Ci) <= 'z')) {
                    Ci++;
                } else {
                    condition = false;
                }
                if (l - Ci < 0) {
                    condition = false;
                }
            }
            for (int Gi = 0; Gi <= l - Ci; Gi++) {
                Updated += exp.charAt(Gi);
            }
            exp = Updated;
        }
        if (exp.length() > 0) {
            if (exp.charAt(0) == '(' && exp.charAt(exp.length() - 1) != ')') {
                exp = FunctionParser.removeHeadAndTail(exp);
            }
            if (exp.charAt(exp.length() - 1) == ')' && exp.charAt(0) != '(') {
                exp = FunctionParser.removeHeadAndTail(exp);
            }
        }
        exp = exp.trim();
        return exp;
    }

    public static String unpackAndCorrect(String exp) {
        if (exp == null) {
            return null;
        }
        if (exp.length() == 0) {
            return "";
        }
        if (exp.equals("()")) {
            return "";
        }
        exp = exp.replace("sigmoid", Function.TYPES.REGISTER[1]);
        exp = exp.replace("quadratic", Function.TYPES.REGISTER[3]);
        exp = exp.replace("quadr", Function.TYPES.REGISTER[3]);
        exp = exp.replace("lig", Function.TYPES.REGISTER[4]);
        exp = exp.replace("ligmoid", Function.TYPES.REGISTER[4]);
        exp = exp.replace("softplus", Function.TYPES.REGISTER[4]);
        exp = exp.replace("spls", Function.TYPES.REGISTER[4]);
        exp = exp.replace("ligm", Function.TYPES.REGISTER[4]);
        exp = exp.replace("identity", Function.TYPES.REGISTER[5]);
        exp = exp.replace("ident", Function.TYPES.REGISTER[5]);
        exp = exp.replace("self", Function.TYPES.REGISTER[5]);
        exp = exp.replace("copy", Function.TYPES.REGISTER[5]);
        exp = exp.replace("gaussian", Function.TYPES.REGISTER[6]);
        exp = exp.replace("gauss", Function.TYPES.REGISTER[6]);
        exp = exp.replace("absolute", Function.TYPES.REGISTER[7]);
        exp = exp.replace("summation", Function.TYPES.REGISTER[10]);
        exp = exp.replace("product", Function.TYPES.REGISTER[11]);

        int bracketDepth = 0;
        for (int Ei = 0; Ei < exp.length(); ++Ei) {
            if (exp.charAt(Ei) == ')') {
                --bracketDepth;
            } else if (exp.charAt(Ei) == '(') {
                ++bracketDepth;
            }
        }
        if (bracketDepth != 0) {
            if (bracketDepth < 0) {
                for (int Bi = 0; Bi < -bracketDepth; ++Bi) {
                    exp = "(" + exp;
                }
            } else if (bracketDepth > 0) {
                for (int Bi = 0; Bi < bracketDepth; ++Bi) {
                    exp = exp + ")";
                }
            }
        }

        boolean parsing = true;
        boolean needsStitching = false;
        while (parsing && exp.charAt(0) == '(' && exp.charAt(exp.length() - 1) == ')') {
            bracketDepth = 0;
            needsStitching = true;
            for (int i = 0; i < exp.length(); ++i) {
                if (exp.charAt(i) == ')') {
                    --bracketDepth;
                } else if (exp.charAt(i) == '(') {
                    ++bracketDepth;
                }
                if (bracketDepth == 0 && i != exp.length() - 1) {
                    needsStitching = false;
                }
            }
            if (needsStitching) {
                exp = FunctionParser.removeHeadAndTail(exp);
            } else {
                parsing = false;
            }
        }
        return exp;
    }

    public static double similarity(String s1, String s2) {
        String longer = (s1.length()>s2.length())?s1:s2, shorter = (s1.length()>s2.length())?s2:s1;
        if (s1.length() < s2.length()) { // longer should always have greater length
            longer = s2; shorter = s1;
        }
        int longerLength = longer.length();
        if (longerLength == 0) { return 1.0; /* both strings are zero length */ }
        return (longerLength - editDistance(longer, shorter)) / (double) longerLength;
    }

    public static int editDistance(String s1, String s2) {
        s1 = s1.toLowerCase();
        s2 = s2.toLowerCase();
        int[] costs = new int[s2.length() + 1];
        for (int i = 0; i <= s1.length(); i++) {
            int lastValue = i;
            for (int j = 0; j <= s2.length(); j++) {
                if (i == 0)
                    costs[j] = j;
                else {
                    if (j > 0) {
                        int newValue = costs[j - 1];
                        if (s1.charAt(i - 1) != s2.charAt(j - 1))
                            newValue = Math.min(Math.min(newValue, lastValue),
                                    costs[j]) + 1;
                        costs[j - 1] = lastValue;
                        lastValue = newValue;
                    }
                }
            }
            if (i > 0)
                costs[s2.length()] = lastValue;
        }
        return costs[s2.length()];
    }

}
