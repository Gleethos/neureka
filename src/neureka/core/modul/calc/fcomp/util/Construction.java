package neureka.core.modul.calc.fcomp.util;

import neureka.core.T;
import neureka.core.modul.calc.FunctionFactory;
import neureka.core.modul.calc.fcomp.Constant;
import neureka.core.modul.calc.fcomp.Function;
import neureka.core.modul.calc.fcomp.Input;
import neureka.core.modul.calc.fcomp.Variable;

import java.util.ArrayList;
import java.util.function.Supplier;

public class Construction {

    public static Function createFunction(int f_id, ArrayList<Function> Srcs, boolean tipReached) {
        boolean[] isFlat = {false};//=> !isFlat
        Srcs.forEach((v) -> {
            isFlat[0] = (
                    (!(v instanceof Input)) &&
                            (!(v instanceof Variable)) &&
                            (!(v instanceof Constant))
            )
                    || isFlat[0];
        });
        isFlat[0] = !isFlat[0];

        Supplier<String> string = () -> {
            String reconstructed = "";
            if (Srcs.size() == 1 && Context.register[f_id].length() > 1) {
                String expression = Srcs.get(0).toString();
                if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                    return Context.register[f_id] + expression;
                }
                return Context.register[f_id] + "(" + expression + ")";
            }
            for (int i = 0; i < Srcs.size(); ++i) {
                if (Srcs.get(i) != null) {
                    reconstructed = reconstructed + Srcs.get(i).toString();
                } else {
                    reconstructed = reconstructed + "(null)";
                }
                if (i != Srcs.size() - 1) {
                    reconstructed = reconstructed + Context.register[f_id];
                }
            }
            return "(" + reconstructed + ")";
        };

        if (f_id < 9) {// FUNCTIONS:
            return new Function() {
                @Override
                public boolean isFlat() {
                    return isFlat[0];
                }

                @Override
                public int id() {
                    return f_id;
                }

                @Override
                public Function newBuild(String expression) {
                    return new FunctionFactory().newBuild(expression);
                }

                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                @Override
                public String toString() {
                    return string.get();//Context.register[f_id];
                }

                @Override
                public T activate(T[] input, int j) {
                    return Calculation.tensorActivationOf(Srcs.get(0).activate(input, j), f_id, false, tipReached, isFlat[0]);
                }

                @Override
                public T activate(T[] input) {
                    return Calculation.tensorActivationOf(Srcs.get(0).activate(input), f_id, false, tipReached, isFlat[0]);
                }

                @Override
                public T derive(T[] input, int d, int j) {
                    return Calculation.tensorActivationOf(Srcs.get(0).activate(input, j), f_id, true, tipReached, isFlat[0]);
                }

                @Override
                public T derive(T[] input, int d) {
                    return Calculation.tensorActivationOf(Srcs.get(0).activate(input), f_id, true, tipReached, isFlat[0]);
                }

                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                @Override
                public double activate(final double[] input, int j) {
                    return Calculation.scalarActivationOf(Srcs.get(0).activate(input, j), f_id, false);
                }

                @Override
                public double activate(final double[] input) {
                    return Calculation.scalarActivationOf(Srcs.get(0).activate(input), f_id, false);
                }

                @Override
                public double derive(final double[] input, final int index, final int j) {
                    return Calculation.scalarActivationOf(Srcs.get(0).activate(input, j), f_id, true)
                            * Srcs.get(0).derive(input, index, j);
                }

                @Override
                public double derive(final double[] input, final int index) {
                    return Calculation.scalarActivationOf(Srcs.get(0).activate(input), f_id, true)
                            * Srcs.get(0).derive(input, index);
                }
            };
        } else {
            return new Function() {
                @Override
                public boolean isFlat() {
                    return isFlat[0];
                }

                @Override
                public int id() {
                    return f_id;
                }

                @Override
                public Function newBuild(String expression) {
                    return new FunctionFactory().newBuild(expression);
                }

                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                @Override
                public String toString() {
                    return string.get();
                }

                @Override
                public T activate(T[] input, int j) {
                    return Calculation.tensorActivationOf(input, f_id, j, -1, Srcs, tipReached, isFlat[0]);
                }

                @Override
                public T activate(T[] input) {
                    return Calculation.tensorActivationOf(input, f_id, -1, -1, Srcs, tipReached, isFlat[0]);
                }

                @Override
                public T derive(T[] input, int d, int j) {
                    return Calculation.tensorActivationOf(input, f_id, j, d, Srcs, tipReached, isFlat[0]);
                }

                @Override
                public T derive(T[] input, int d) {
                    return Calculation.tensorActivationOf(input, f_id, -1, d, Srcs, tipReached, isFlat[0]);
                }

                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                @Override
                public double activate(final double[] input, int j) {
                    return Calculation.scalarActivationOf(input, f_id, j, -1, Srcs);
                }

                @Override
                public double activate(final double[] input) {
                    return Calculation.scalarActivationOf(input, f_id, -1, -1, Srcs);
                }

                @Override
                public double derive(final double[] input, final int d, final int j) {
                    return Calculation.scalarActivationOf(input, f_id, j, d, Srcs);
                }

                @Override
                public double derive(final double[] input, final int d) {
                    return Calculation.scalarActivationOf(input, f_id, -1, d, Srcs);
                }
            };

        }
    }
}
/**
 * case 0:  return "relu";
 * case 1:  return "sig";
 * case 2:  return "tanh";
 * case 3:  return "quad";
 * case 4:  return "lig";
 * case 5:  return "lin";
 * case 6:  return "gaus";
 * case 7:  return "abs";
 * case 8:  return "sin";
 * case 9:  return "cos";
 * <p>
 * case 10: return "sum";
 * case 11: return "prod";
 * <p>
 * case 12: return "^";
 * case 13: return "/";
 * case 14: return "*";
 * case 15: return "%";
 * case 16: return "-";
 * case 17: return "+";
 * <p>
 * case 18: return "x"; (conv/tm)
 */
