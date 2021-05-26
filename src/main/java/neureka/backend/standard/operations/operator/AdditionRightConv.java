package neureka.backend.standard.operations.operator;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.calculus.Function;

public class AdditionRightConv extends AbstractOperation {

    public AdditionRightConv() {
        super(
                new OperationBuilder()
                        .setFunction(         ""                 )
                        .setOperator(         "a" + ((char) 187) )
                        .setArity(            3                  )
                        .setIsOperator(       true               )
                        .setIsIndexer(        false              )
                        .setIsDifferentiable( false              )
                        .setIsInline(         false              )
        );
    }

    @Override
    public String stringify(String[] children) {
        return null;
    }

    @Override
    public String asDerivative(Function[] children, int d ) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }

}