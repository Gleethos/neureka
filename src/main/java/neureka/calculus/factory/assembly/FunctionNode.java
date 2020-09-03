package neureka.calculus.factory.assembly;

import neureka.Tsr;
import neureka.calculus.*;
import neureka.calculus.environment.OperationType;
import neureka.calculus.factory.AbstractFunction;

import java.util.List;

public class FunctionNode extends AbstractFunction
{
    public FunctionNode(OperationType type, List<Function> sources, boolean doAD ) {
        super( type, sources, doAD );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public Tsr call( Tsr[] inputs, int j ) {
        return CACHE.preprocess( inputs, this, ()-> _tensor_activation(inputs, j, -1), -1, j );
    }

    @Override
    public Tsr call( Tsr[] inputs ) {
        return CACHE.preprocess( inputs, this, ()-> _tensor_activation(inputs, -1, -1), -1, -1 );
    }

    @Override
    public Tsr derive( Tsr[] inputs, int d, int j ) {
        return CACHE.preprocess( inputs, this, ()-> _tensor_activation(inputs, j, d), d, j );
    }

    @Override
    public Tsr derive( Tsr[] inputs, int d ) {
        return CACHE.preprocess( inputs, this, ()-> _tensor_activation(inputs, -1, d), d, -1 );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public double call( final double[] inputs, int j ) {
        return this.type().calculate( inputs, j, -1, this.getChildren() );
    }

    @Override
    public double call( final double[] inputs ) {
        return this.type().calculate( inputs, -1, -1, this.getChildren() );
    }

    @Override
    public double derive( final double[] inputs, final int d, final int j ) {
        return this.type().calculate( inputs, j, d, this.getChildren() );
    }

    @Override
    public double derive( final double[] inputs, final int d ) {
        return this.type().calculate( inputs, -1, d, this.getChildren() );
    }

}