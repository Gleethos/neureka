package neureka.backend.main.operations.other;

import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.ndim.config.NDConfiguration;

public class ReLayout extends AbstractOperation
{
    public ReLayout()
    {
        super(
                new OperationBuilder()
                        .identifier(       "layout"  )
                        .operator(         "layout"  )
                        .arity(            1          )
                        .isOperator(       false      )
                        .isIndexer(        false      )
                        .isDifferentiable( true       )
                        .isInline(         false      )
        );
        setAlgorithm(
                Algorithm
                        .withName( "layout" )
                        .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
                        .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
                        .setExecution(
                                ( caller, call ) ->
                                {
                                    Tsr<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
                                    Tsr<Object> input = (Tsr<Object>) inputs[0];

                                    NDConfiguration.Layout originalLayout = input.getNDConf().getLayout();
                                    NDConfiguration.Layout newLayout = call.getValOf( Arg.Layout.class );

                                    Tsr<?> reLayout = input.deepCopy().mut().toLayout(newLayout);

                                    return Result.of(reLayout.mut().setIsIntermediate(true))
                                            .withADAction( target -> {
                                                Tsr<Object> error = (Tsr<Object>) target.error().deepCopy();
                                                return error.mut().toLayout(originalLayout);
                                            });
                                }
                        )
                        .buildFunAlgorithm()
        );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
