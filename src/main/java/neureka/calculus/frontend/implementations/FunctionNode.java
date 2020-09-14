package neureka.calculus.frontend.implementations;

import neureka.Tsr;
import neureka.calculus.*;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.frontend.AbstractFunction;

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
        return this.getOperation().calculate( inputs, j, -1, this.getChildren() );
    }

    @Override
    public double call( final double[] inputs ) {
        return this.getOperation().calculate( inputs, -1, -1, this.getChildren() );
    }

    @Override
    public double derive( final double[] inputs, final int d, final int j ) {
        return this.getOperation().calculate( inputs, j, d, this.getChildren() );
    }

    @Override
    public double derive( final double[] inputs, final int d ) {
        return this.getOperation().calculate( inputs, -1, d, this.getChildren() );
    }

}