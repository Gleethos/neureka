package neureka.core.module.calc.fcomp.util;

import neureka.core.T;
import neureka.core.module.calc.fcomp.*;

import java.util.ArrayList;

public class Construction {

    public static Function createFunction(int f_id, ArrayList<Function> Srcs){
        boolean[] isFlat = {false};//=> !isFlat
        Srcs.forEach((v)->{
            isFlat[0] = (
                    (!(v instanceof Input)) && (!(v instanceof Variable)) && (!(v instanceof Constant))
                    ) || isFlat[0];
        });
        isFlat[0] = !isFlat[0];
        if(f_id<9) {// FUNCTIONS:
            return new Template(f_id, isFlat[0], Srcs){//Function(){
                @Override
                public T activate(T[] input, int j) {
                    return Calculation.tensorActivationOf(Srcs.get(0).activate(input, j), f_id, false, isFlat);
                }
                @Override
                public T activate(T[] input) {
                    return Calculation.tensorActivationOf(Srcs.get(0).activate(input), f_id, false, isFlat);
                }
                @Override
                public T derive(T[] input, int d, int j) {
                    return Calculation.tensorActivationOf(Srcs.get(0).activate(input, j), f_id, true, isFlat);
                }
                @Override
                public T derive(T[] input, int d) {
                    return Calculation.tensorActivationOf(Srcs.get(0).activate(input), f_id, true, isFlat);
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
        }else{
            return new Template(f_id, isFlat[0], Srcs){//Function(){
                @Override
                public T activate(T[] input, int j) {
                    return Calculation.tensorActivationOf(input, f_id, j, -1, Srcs, isFlat);
                }
                @Override
                public T activate(T[] input) {
                    return Calculation.tensorActivationOf(input, f_id, -1, -1, Srcs, isFlat);
                }
                @Override
                public T derive(T[] input, int d, int j) {
                    return Calculation.tensorActivationOf(input, f_id, j, d, Srcs, isFlat);
                }
                @Override
                public T derive(T[] input, int d) {
                    return Calculation.tensorActivationOf(input, f_id, -1, d, Srcs, isFlat);
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
    case 0:  return "relu";
	case 1:  return "sig";
	case 2:  return "tanh";
	case 3:  return "quad";
	case 4:  return "lig";
	case 5:  return "lin";
	case 6:  return "gaus";
	case 7:  return "abs";
	case 8:  return "sin";
	case 9:  return "cos";

	case 10: return "sum";
	case 11: return "prod";

	case 12: return "^";
	case 13: return "/";
	case 14: return "*";
	case 15: return "%";
	case 16: return "-";
	case 17: return "+";

    case 18: return "x"; (conv/tm)
 * */
