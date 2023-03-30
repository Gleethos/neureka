package neureka.backend.main.operations.other;

import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;

public class DimFit extends AbstractOperation
{
    public DimFit()
    {
        super(
            new OperationBuilder()
                    .identifier(       "dimfit"    )
                    .operator(         "dimfit"    )
                    .arity(            -1          )
                    .isOperator(       false       )
                    .isIndexer(        false       )
                    .isDifferentiable( true        )
                    .isInline(         false       )
        );

        setAlgorithm(
            Algorithm
            .withName("dimFit")
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    assert call.getValOf( Arg.DerivIdx.class ) < 0;
                    Tsr<?>[] inputs = AbstractDeviceAlgorithm.flatten( caller, call ).inputs();

                    int largest = -1;
                    int[] shape = null;
                    for ( Tsr<?> t : inputs ) if ( t.rank() > largest ) {
                        largest = t.rank();
                        shape = t.getNDConf().shape();
                    }
                    int prefix = 0;
                    for ( int s : shape ) if ( s == 1 ) prefix++; else break;
                    int postfix = 0;
                    for ( int i = shape.length-1; i>=0; i-- ) if ( shape[ i ] == 1 ) postfix++; else break;

                    int[][] change = new int[inputs.length][];

                    for ( int i=0; i<inputs.length; i++)
                    {
                        if ( inputs[ i ].rank()!=largest)
                        {
                            int[] oldShape = inputs[ i ].getNDConf().shape();
                            int[] newReshape = new int[largest];
                            int padding = largest-oldShape.length;

                            int handle = ( postfix <= prefix )? padding : largest-padding;
                            for ( int ii = 0; ii < handle; ii++ ) newReshape[ ii ]      = ( postfix <= prefix )? -1 : ii;
                            for ( int ii = handle; ii < largest; ii++) newReshape[ ii ] = ( postfix <= prefix )? ii-padding : -1;

                            change[ i ] = newReshape;
                        }
                    }
                    return Result.of(null).withADAction(null);
                }
            )
            .buildFunAlgorithm()
        );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }
}
