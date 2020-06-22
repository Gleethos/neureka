package neureka.calculus.factory.assembly;

import neureka.Tsr;
import neureka.calculus.*;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.Type;
import neureka.calculus.factory.AbstractFunction;
import neureka.calculus.factory.components.FunctionConstant;
import neureka.calculus.factory.components.FunctionInput;
import neureka.calculus.factory.components.FunctionVariable;

import java.util.ArrayList;

public class FunctionConstructor
{
    public static Function construct(int f_id, ArrayList<Function> sources, boolean doAD)
    {
        Type type = OperationType.instance(f_id);
        if(type.numberOfParameters() >= 0 && sources.size() != type.numberOfParameters()) {
            String tip = (type.isIndexer())?
            "\nNote: This function is an 'indexer'. Therefore it expects to sum variable 'I[j]' inputs, where 'j' is the index of an iteration.":"";
            throw new IllegalArgumentException(
                    "The function/operation '"+type.identifier()+"' expects "+type.numberOfParameters()+" parameters, "+
                    "however "+sources.size()+" where given!"+tip
            );

        }
        boolean isFlat = true;
        for (Function f : sources) {// AbstractFunction does only reference tip nodes of the function graph:
            isFlat = ((f instanceof FunctionInput) || (f instanceof FunctionVariable) || (f instanceof FunctionConstant)) && isFlat;
        }
        if ( f_id <= 9 ) {// FUNCTIONS:
            return new AbstractFunction(f_id, isFlat, sources, doAD){
                @Override
                public Tsr call(Tsr[] inputs, int j) {
                    return CACHE.preprocess(inputs, this,()-> _tensor_activation(new Tsr[]{sources.get(0).call(inputs, j)}, j, -1), -1, j);
                }
                @Override
                public Tsr call(Tsr[] inputs) {
                    return CACHE.preprocess(inputs, this,()-> _tensor_activation(new Tsr[]{sources.get(0).call(inputs)}, -1, -1), -1, -1);
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
                public double call(final double[] inputs, int j) {
                    return _scalar_activation(sources.get(0).call(inputs, j), false);
                }
                @Override
                public double call(final double[] inputs) {
                    return _scalar_activation(sources.get(0).call(inputs), false);
                }
                @Override
                public double derive(final double[] inputs, final int index, final int j) {
                    return _scalar_activation(sources.get(0).call(inputs, j), true)
                            * sources.get(0).derive(inputs, index, j);
                }
                @Override
                public double derive(final double[] inputs, final int index) {
                    return _scalar_activation(sources.get(0).call(inputs), true)
                            * sources.get(0).derive(inputs, index);
                }
            };
        } else {
            return new AbstractFunction(f_id, isFlat, sources, doAD) {
                @Override
                public Tsr call(Tsr[] inputs, int j) {
                    return CACHE.preprocess(inputs, this, ()-> _tensor_activation(inputs, j, -1), -1, j);
                }
                @Override
                public Tsr call(Tsr[] inputs) {
                    return CACHE.preprocess(inputs, this, ()-> _tensor_activation(inputs, -1, -1), -1, -1);
                }
                @Override
                public Tsr derive(Tsr[] inputs, int d, int j) {
                    return CACHE.preprocess(inputs, this, ()-> _tensor_activation(inputs, j, d), d, j);
                }
                @Override
                public Tsr derive(Tsr[] inputs, int d) {
                    return CACHE.preprocess(inputs, this, ()-> _tensor_activation(inputs, -1, d), d, -1);
                }
                //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                @Override
                public double call(final double[] inputs, int j) {
                    return _scalar_activation(inputs, j, -1);
                }
                @Override
                public double call(final double[] inputs) {
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