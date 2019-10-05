package neureka.core.function.factory.assembly;

import neureka.core.Tsr;
import neureka.core.function.*;
import neureka.core.function.factory.implementations.FConstant;
import neureka.core.function.factory.implementations.FInput;
import neureka.core.function.factory.Function;
import neureka.core.function.factory.implementations.FVariable;

import java.util.ArrayList;

public class FunctionConstructor
{
    public static IFunction construct(int f_id, ArrayList<IFunction> sources, boolean doAD)
    {
        boolean isFlat = true;
        for(IFunction f : sources){// Function does only reference tip nodes of the function graph:
            isFlat = ((f instanceof FInput) || (f instanceof FVariable) || (f instanceof FConstant)) && isFlat;
        }
        if(f_id<=9) {// FUNCTIONS:
            return new Function(f_id, isFlat, sources, doAD){
                @Override
                public Tsr activate(Tsr[] inputs, int j) {
                    return CACHE.handle(inputs, this,()-> _tensor_activation(sources.get(0).activate(inputs, j), false));
                }
                @Override
                public Tsr activate(Tsr[] inputs) {
                    return CACHE.handle(inputs, this, ()-> _tensor_activation(sources.get(0).activate(inputs), false));
                }
                @Override
                public Tsr derive(Tsr[] inputs, int d, int j) {
                    return _tensor_activation(sources.get(0).activate(inputs, j), true);
                }
                @Override
                public Tsr derive(Tsr[] inputs, int d) {
                    return _tensor_activation(sources.get(0).activate(inputs), true);
                }
                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                @Override
                public double activate(final double[] inputs, int j) {
                    return _scalar_activation(sources.get(0).activate(inputs, j), false);
                }
                @Override
                public double activate(final double[] inputs) {
                    return _scalar_activation(sources.get(0).activate(inputs), false);
                }
                @Override
                public double derive(final double[] inputs, final int index, final int j) {
                    return _scalar_activation(sources.get(0).activate(inputs, j), true)
                            * sources.get(0).derive(inputs, index, j);
                }
                @Override
                public double derive(final double[] inputs, final int index) {
                    return _scalar_activation(sources.get(0).activate(inputs), true)
                            * sources.get(0).derive(inputs, index);
                }
            };
        }else{
            return new Function(f_id, isFlat, sources, doAD){
                @Override
                public Tsr activate(Tsr[] inputs, int j) {
                    return CACHE.handle(inputs, this, ()-> _tensor_activation(inputs, j, -1));
                }
                @Override
                public Tsr activate(Tsr[] inputs) {
                    return CACHE.handle(inputs, this, ()-> _tensor_activation(inputs, -1, -1));
                }
                @Override
                public Tsr derive(Tsr[] inputs, int d, int j) {
                    return _tensor_activation(inputs, j, d);
                }
                @Override
                public Tsr derive(Tsr[] inputs, int d) {
                    return _tensor_activation(inputs, -1, d);
                }
                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                @Override
                public double activate(final double[] inputs, int j) {
                    return _scalar_activation(inputs, j, -1);
                }
                @Override
                public double activate(final double[] inputs) {
                    return _scalar_activation(inputs, -1, -1);
                }
                @Override
                public double derive(final double[] inputs, final int d, final int j) {
                    return _scalar_activation(inputs, j, d);
                }
                @Override
                public double derive(final double[] inputs, final int d) {
                    return _scalar_activation(inputs, -1, d);
                }
            };
        }
    }
}