package neureka.backend.main.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.backend.main.implementations.broadcast.CLBroadcastAddition;
import neureka.backend.main.implementations.broadcast.CPUBroadcastSummation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

/**
 *  This type of operation belongs to the same species as the
 *  {@link Product} operation.
 *  It executes incoming calls so that the calling function
 *  will be executed with all input indices passed to it.
 *  The resulting array of tensors will then be summed
 *  to produce the result of this operation, hence the name {@link Summation}.
 */
public final class Summation extends AbstractOperation
{
    public Summation()
    {
        super (
                new OperationBuilder()
                        .identifier(        "sumJs"    )
                        .operator(          "sumJs"    )
                        .arity(            1           )
                        .isOperator(       false       )
                        .isIndexer(        true        )
                        .isDifferentiable( true        )
                        .isInline(         false       )
        );

        //________________
        // BROADCASTING :

        Broadcast operationAlgorithm = new Broadcast(ElemWiseUtil::forAdditions)
                .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                    {
                        if ( call.autogradMode().allowsForward() )
                            throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        Function mul = Neureka.get().backend().getFunction().mul();
                        if ( ctxDerivative != null ) {
                            return ADAgent.of( target -> mul.execute( target.error(), ctxDerivative ) );
                        }
                        int d = call.getValOf( Arg.DerivIdx.class );
                        Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                        return ADAgent.of( target -> mul.execute( target.error(), derivative ) );
                    }
                )
                .buildFunAlgorithm();


        setAlgorithm(
                Broadcast.class,
                operationAlgorithm.setImplementationFor(
                    CPU.class,
                    new CPUBroadcastSummation()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    new CLBroadcastAddition( this.getIdentifier() )
                )
        );

    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) return _calculate( inputs, src );
        else return src[ 0 ].derive( inputs, d, j );
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 )
            return _calculate( inputs, src );
        else {
            double sum = 0;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                double r = src[ 0 ].derive( inputs, d, i );
                sum += r;
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs );
            return sum;
        }

    }

    private static double _calculate( double[] inputs, Function[] src ) {
        double sum = 0;
        boolean nothingDone = true;
        for ( int i = 0; i < inputs.length; i++ ) {
            sum += src[ 0 ].call( inputs, i );
            nothingDone = false;
        }
        if ( nothingDone ) return src[ 0 ].call( inputs );
        return sum;
    }


}
