package neureka.core.function.factory.assembly;

import neureka.core.T;
import neureka.core.function.*;
import neureka.core.function.factory.implementations.FConstant;
import neureka.core.function.factory.implementations.FInput;
import neureka.core.function.factory.Function;
import neureka.core.function.factory.implementations.FVariable;

import java.util.ArrayList;

public class FunctionConstructor
{
    public static IFunction createFunction(int f_id, ArrayList<IFunction> sources, boolean doAD)
    {
        boolean isFlat = true;
        for(IFunction f : sources){
            isFlat = ((f instanceof FInput) || (f instanceof FVariable) || (f instanceof FConstant)) && isFlat;
        }
        if(f_id<=9) {// FUNCTIONS:
            return new Function(f_id, isFlat, sources, doAD){
                @Override
                public T activate(T[] input, int j) {
                    return T_CACHE.handle(input, this,()->tensorActivationOf(sources.get(0).activate(input, j), false));
                }
                @Override
                public T activate(T[] input) {
                    return T_CACHE.handle(input, this, ()->tensorActivationOf(sources.get(0).activate(input), false));
                }
                @Override
                public T derive(T[] input, int d, int j) {
                    return tensorActivationOf(sources.get(0).activate(input, j), true);
                }
                @Override
                public T derive(T[] input, int d) {
                    return tensorActivationOf(sources.get(0).activate(input), true);
                }
                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                @Override
                public double activate(final double[] input, int j) {
                    return scalarActivationOf(sources.get(0).activate(input, j), false);
                }
                @Override
                public double activate(final double[] input) {
                    return scalarActivationOf(sources.get(0).activate(input), false);
                }
                @Override
                public double derive(final double[] input, final int index, final int j) {
                    return scalarActivationOf(sources.get(0).activate(input, j), true) * sources.get(0).derive(input, index, j);
                }
                @Override
                public double derive(final double[] input, final int index) {
                    return scalarActivationOf(sources.get(0).activate(input), true) * sources.get(0).derive(input, index);
                }
            };
        }else{
            return new Function(f_id, isFlat, sources, doAD){
                @Override
                public T activate(T[] input, int j) {
                    return T_CACHE.handle(input, this, ()->tensorActivationOf(input, j, -1));
                }
                @Override
                public T activate(T[] input) {
                    return T_CACHE.handle(input, this, ()->tensorActivationOf(input, -1, -1));
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
                    return scalarActivationOf(input, j, -1);
                }
                @Override
                public double activate(final double[] input) {
                    return scalarActivationOf(input, -1, -1);
                }
                @Override
                public double derive(final double[] input, final int d, final int j) {
                    return scalarActivationOf(input, j, d);
                }
                @Override
                public double derive(final double[] input, final int d) {
                    return scalarActivationOf(input, -1, d);
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