package neureka.core.module.calc.fcomp.util;

import neureka.core.T;
import neureka.core.module.calc.fcomp.*;

import java.util.ArrayList;

public class Construction {

    public static Function createFunction(int f_id, ArrayList<Function> Srcs, boolean doAD){
        boolean[] isFlat = {false};//=> !isFlat
        Srcs.forEach((v)->{
            isFlat[0] = (
                    (!(v instanceof Input)) && (!(v instanceof Variable)) && (!(v instanceof Constant))
                    ) || isFlat[0];
        });
        isFlat[0] = !isFlat[0];
        if(f_id<9) {// FUNCTIONS:
            return new Template(f_id, isFlat[0], Srcs, doAD){//Function(){
                @Override
                public T activate(T[] input, int j) {
                    return tensorActivationOf(Srcs.get(0).activate(input, j), false);
                }
                @Override
                public T activate(T[] input) {
                    return tensorActivationOf(Srcs.get(0).activate(input), false);
                }
                @Override
                public T derive(T[] input, int d, int j) {
                    return tensorActivationOf(Srcs.get(0).activate(input, j), true);
                }
                @Override
                public T derive(T[] input, int d) {
                    return tensorActivationOf(Srcs.get(0).activate(input), true);
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
            return new Template(f_id, isFlat[0], Srcs, doAD){//Function(){
                @Override
                public T activate(T[] input, int j) {
                    return tensorActivationOf(input, j, -1);
                }
                @Override
                public T activate(T[] input) {
                    return tensorActivationOf(input, -1, -1);
                }
                @Override
                public T derive(T[] input, int d, int j) {
                    return tensorActivationOf(input, j, d);
                }
                @Override
                public T derive(T[] input, int d) {
                    return tensorActivationOf(input, -1, d);
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
    case 0:  "relu";
	case 1:  "sig";
	case 2:  "tanh";
	case 3:  "quad";
	case 4:  "lig";
	case 5:  "lin";
	case 6:  "gaus";
	case 7:  "abs";
	case 8:  "sin";
	case 9:  "cos";

	case 10: "sum";
	case 11: "prod";

	case 12: "^";
	case 13: "/";
	case 14: "*";
	case 15: "%";
	case 16: "-";
	case 17: "+";
    case 18: "x"; (conv/tm)
 * */
