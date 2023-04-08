package neureka.backend.main.operations.other;

import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.framing.Relation;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;

public class Reshape extends AbstractOperation
{
    public Reshape()
    {
        super(
            new OperationBuilder()
                .identifier(       "reshape"  )
                .operator(         "reshape"  )
                .arity(            1          )
                .isOperator(       false      )
                .isIndexer(        false      )
                .isDifferentiable( true       )
                .isInline(         false      )
        );
        setAlgorithm(
            Algorithm
            .withName( "reshape" )
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    Tsr<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
                    Tsr<Object> input = (Tsr<Object>) inputs[0];

                    int[] shape = call.getValOf( Arg.Shape.class );

                    if ( shape == null )
                        throw new IllegalArgumentException("Shape argument is missing!");

                    Tsr reshaped = Tsr.of(
                                    input.getDataType(),
                                    NDConstructor.of( shape ),
                                    input.mut().getData()
                                );

                    reshaped.set( Relation.newChildToParent( input ) );
                    Relation parent = input.find( Relation.class ).orElseGet(Relation::newParentToChildren);
                    parent.addChild( reshaped );
                    input.set( parent );

                    if ( input.isOutsourced() )
                        input.getDevice().store( reshaped );

                    NDConfiguration originalConfig = input.getNDConf();

                    return Result.of(reshaped.mut().setIsIntermediate(true))
                            .withADAction( target -> {
                                Tsr<Object> error = (Tsr<Object>) target.error();
                                return Tsr.of(
                                        error.getDataType(),
                                        NDConstructor.of( originalConfig ),
                                        error.mut().getData()
                                    );
                            });
                }
            )
            .buildFunAlgorithm()
        );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src )
    {
        return src[ 0 ].call( inputs, j );
    }
}
