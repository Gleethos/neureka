package neureka.main.core.modul.calc;

import java.util.List;
import java.util.ListIterator;

public class NFUtility {

    //public static boolean isBasicOperation(final String operation) {
    //    if (operation.length() > 8) {
    //        return false;
    //    }
    //    for (int i = 0; i < FunctionFactory.OperationRegister.length; ++i) {
    //        System.out.print(FunctionFactory.OperationRegister[i] + " =?= " + operation + " -:|:- ");
    //        if (FunctionFactory.OperationRegister[i].equals(operation)) {
    //            System.out.println("");
    //            return true;
    //        }
    //    }
    //    return false;
    //}
//
//
    //public static String removeHeadAndTail(final String equation) {
    //    String corrected = "";
    //    for (int i = 1; i < equation.length() - 1; ++i) {
    //        corrected = String.valueOf(corrected) + equation.charAt(i);
    //    }
    //    return corrected;
    //}
//
    //public static String parsedComponent(final String equation, final int i) {
    //    if (equation.length() <= i) {
    //        return null;
    //    }
    //    String component = "";
    //    int bracketDepth = 0;
    //    component = "";
    //    System.out.print("Start char: " + equation.charAt(i) + "\n");
    //    for (int Ei = i; Ei < equation.length(); ++Ei) {
    //        if (equation.charAt(Ei) == ')') {
    //            --bracketDepth;
    //        } else if (equation.charAt(Ei) == '(') {
    //            ++bracketDepth;
    //        }
    //        System.out.print("d[" + bracketDepth + "]:[  " + equation.charAt(Ei) + "  ], ");
    //        if (bracketDepth == 0) {
    //            String possibleOperation = "";
    //            for (int Sii = Ei + 1; Sii < equation.length(); ++Sii) {
    //                possibleOperation = String.valueOf(possibleOperation) + equation.charAt(Sii);
    //                if (isBasicOperation(possibleOperation)) {
    //                    component = String.valueOf(component) + equation.charAt(Ei);
    //                    System.out.print("\n");
    //                    return component;
    //                }
    //            }
    //        }
    //        component = String.valueOf(component) + equation.charAt(Ei);
    //    }
    //    System.out.print("\n");
    //    return component;
    //}
//
    //public static int numberOfOperationsWithin(final List<String> operations) {
    //    int Count = 0;
    //    for (int i = 0; i < FunctionFactory.OperationRegister.length; ++i) {
    //        if (NFUtility.containsOperation(FunctionFactory.OperationRegister[i], operations)) {
    //            ++Count;
    //        }
    //    }
    //    return Count;
    //}
//
    //public static boolean containsOperation(final String operation, final List<String> operations) {
    //    final ListIterator<String> OperationIterator = operations.listIterator();
    //    while (OperationIterator.hasNext()) {
    //        final String currentOperation = OperationIterator.next();
    //        if (currentOperation.equals(operation)) {
    //            return true;
    //        }
    //    }
    //    return false;
    //}
//
    //public static String unpackAndCorrect(String equation) {
    //    if (equation == null) {
    //        return null;
    //    }
    //    if (equation.length() == 0) {
    //        return "";
    //    }
    //    if (equation.equals("()")) {
    //        return "";
    //    }
    //    equation = equation.replace("lig", FunctionFactory.OperationRegister[4]);
    //    equation = equation.replace("ligmoid", FunctionFactory.OperationRegister[4]);
    //    equation = equation.replace("softplus", FunctionFactory.OperationRegister[4]);
    //    equation = equation.replace("spls", FunctionFactory.OperationRegister[4]);
    //    equation = equation.replace("ligm", FunctionFactory.OperationRegister[4]);
    //    equation = equation.replace("linear", FunctionFactory.OperationRegister[5]);
    //    equation = equation.replace("sigmoid", FunctionFactory.OperationRegister[1]);
    //    equation = equation.replace("quadratic", FunctionFactory.OperationRegister[3]);
    //    equation = equation.replace("quadr", FunctionFactory.OperationRegister[3]);
    //    equation = equation.replace("gaussian", FunctionFactory.OperationRegister[6]);
    //    equation = equation.replace("gauss", FunctionFactory.OperationRegister[6]);
    //    equation = equation.replace("summation", FunctionFactory.OperationRegister[7]);
    //    equation = equation.replace("product", FunctionFactory.OperationRegister[8]);
    //    equation = equation.replace("absolute", FunctionFactory.OperationRegister[9]);
    //    /*
    //     * OCode = new String[]
    //     * {"relu", "sig", "tanh", "quad", "lig", "lin", "gaus",
    //     *  "sum",  "prod",
    //     *  "^",     "/",  "*",     "%",   "+",   "-" };
    //     * */
    //    int bracketDepth = 0;
    //    for (int Ei = 0; Ei < equation.length(); ++Ei) {
    //        if (equation.charAt(Ei) == ')') {
    //            --bracketDepth;
    //        } else if (equation.charAt(Ei) == '(') {
    //            ++bracketDepth;
    //        }
    //    }
    //    if (bracketDepth != 0) {
    //        if (bracketDepth < 0) {
    //            for (int Bi = 0; Bi < -bracketDepth; ++Bi) {
    //                equation = "(" + equation;
    //            }
    //        } else if (bracketDepth > 0) {
    //            for (int Bi = 0; Bi < bracketDepth; ++Bi) {
    //                equation = String.valueOf(equation) + ")";
    //            }
    //        }
    //    }
    //    return equation;
    //}
}
