package neureka.backend.main.operations.other;

import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.ScalarSumAlgorithm;
import neureka.backend.main.algorithms.SumAlgorithm;
import neureka.calculus.Function;

public class Sum extends AbstractOperation
{
    public Sum()
    {
        super(
            new OperationBuilder()
                .identifier(       "sumItems"  )
                .operator(         "sumItems"  )
                .arity(            1           )
                .isOperator(       false       )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );
        setAlgorithm(new ScalarSumAlgorithm());
        setAlgorithm(new SumAlgorithm());
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
