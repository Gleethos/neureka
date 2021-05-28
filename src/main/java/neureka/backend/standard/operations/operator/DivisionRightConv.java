package neureka.backend.standard.operations.operator;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.calculus.Function;

public class DivisionRightConv extends AbstractOperation {

    public DivisionRightConv() {
        super(
                new OperationBuilder()
                        .setFunction(         "div_conv_right"                 )
                        .setOperator(         "d" + ((char) 187) )
                        .setArity(            3                  )
                        .setIsOperator(       true               )
                        .setIsIndexer(        false              )
                        .setIsDifferentiable( true               )
                        .setIsInline(         false              )
        );
    }

    @Override
    public String stringify(String[] children) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" d" + ((char) 187)+" ");
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
        return src[ 0 ].call( inputs, j );
    }

}
