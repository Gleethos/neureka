package neureka.backend.main.operations.linear;

import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.DotProductAlgorithm;
import neureka.math.Function;

public class DotProduct extends AbstractOperation
{
    public DotProduct() {
        super(
            new OperationBuilder()
            .identifier(       "dot"       )
            .operator(         "dot"       )
            .arity(            2           )
            .isOperator(       false       )
            .isIndexer(        false       )
            .isDifferentiable( true        )
            .isInline(         false       )
        );
        setAlgorithm(
            new DotProductAlgorithm().buildFunAlgorithm()
        );
    }

    @Override
    public double calculate(double[] inputs, int j, int d, Function[] src) {
        throw new UnsupportedOperationException();
    }
}
