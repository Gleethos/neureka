package neureka.backend.standard.operations.linear;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.operations.ConvUtil;
import neureka.calculus.Function;

public class XConvRight extends AbstractOperation {

    public XConvRight() {
        super(
                new OperationBuilder()
                        .setFunction(         "inv_convolve_mul_right"    )
                        .setOperator(         "x" + ((char) 187)         )
                        .setArity(            3                         )
                        .setIsOperator(       true        )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( false       )
                        .setIsInline(         false       )
        );
        setAlgorithm( Convolution.class, ConvUtil.getConv() );
    }

    @Override
    public String stringify(String[] children) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" x" + ((char) 187)+" ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return 0;
    }

}
