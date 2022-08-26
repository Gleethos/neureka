package neureka.backend.main.operations.linear;

import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Convolution;
import neureka.backend.main.operations.ConvUtil;
import neureka.calculus.Function;

public class XConvRight extends AbstractOperation {

    public XConvRight() {
        super(
            new OperationBuilder()
                    .identifier(         "inv_convolve_mul_right"    )
                    .operator(         "x" + ((char) 187)         )
                    .arity(            3                         )
                    .isOperator(       true        )
                    .isIndexer(        false       )
                    .isDifferentiable( false       )
                    .isInline(         false       )
        );
        setAlgorithm( Convolution.class, ConvUtil.getConv() );
    }

    @Override
    public String stringify( String[] children ) {
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
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return 0;
    }

}
