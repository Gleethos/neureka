package neureka.backend.main.operations.linear;

import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.MatMulAlgorithm;
import neureka.math.Function;

public class MatMul extends AbstractOperation
{
    public MatMul()
    {
        super(
            new OperationBuilder()
                .identifier(       "matMul"    )
                .operator(         "@"         )
                .arity(            2           )
                .isOperator(       true        )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );

        setAlgorithm(
            new MatMulAlgorithm().buildFunAlgorithm()
        );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
