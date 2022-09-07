package neureka.backend.main.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Convolution;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.backend.main.implementations.scalar.CPUElementwiseActivation;
import neureka.backend.main.functions.ScalarFun;
import neureka.backend.main.operations.operator.impl.CLBroadcastAddition;
import neureka.backend.main.operations.operator.impl.CPUBroadcastSummation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
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
                            return ADAgent.of( ctxDerivative )
                                            .withAD( target -> mul.execute( target.error(), ctxDerivative ) );
                        }
                        int d = call.getValOf( Arg.DerivIdx.class );
                        Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                        return ADAgent.of( derivative )
                                        .withAD( target -> mul.execute( target.error(), derivative ) );
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


        //______________
        // ACTIVATION :

        Activation activation = new Activation()
        .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
        .setDeviceExecution(
            (call, callback) -> ElemWiseUtil.forAdditions(call, callback),
            ( Function f, ExecutionCall<? extends Device<?>> adCall ) ->
            {
                Tsr<?> ctxDerivative = (Tsr<?>) adCall.getValOf(Arg.Derivative.class);
                Function mul = Neureka.get().backend().getFunction().mul();
                if ( ctxDerivative != null )
                    return ADAgent.of( ctxDerivative )
                                    .withAD( target -> mul.execute( target.error(), ctxDerivative ) );

                int d = adCall.getDerivativeIndex();
                if ( adCall.autogradMode().allowsForward() )
                {
                    Tsr<?> derivative = f.executeDerive( adCall.inputs(), d );
                    return ADAgent.of( derivative )
                                    .withAD( target -> mul.execute( target.error(), derivative ) );
                }
                else
                {
                    if ( this.supports(Convolution.class) )
                    {
                        Function deConv = new FunctionParser( Neureka.get().backend() ).parse(
                                "I[ 0 ]" + getOperator() + ">>I[ 1 ]" + getOperator() + ">>I[ 2 ]",
                                false
                        );
                        Tsr<?> derivative = f.executeDerive( adCall.inputs(), d );
                        return ADAgent.of( derivative )
                                .withAD(
                                    adCall.autogradMode() == AutoDiffMode.FORWARD_ONLY
                                    ? target -> mul.execute( target.error(), derivative )
                                    : target ->
                                        deConv.execute(
                                                target.error(),
                                                derivative,
                                                Tsr.of(target.node().getPayload().shape(), 0) ).getUnsafe().setIsIntermediate( true )
                                );
                    }
                    else
                    {
                        Tsr<?> derivative = f.executeDerive( adCall.inputs(), d );
                        return ADAgent.of( derivative )
                                        .withAD( target -> mul.execute( target.error(), derivative ) );
                    }
                }
            }
        )
        .setCallPreparation(
                call -> {
                    Device<Number> device = call.getDeviceFor(Number.class);
                    if ( call.input( 0 ) == null ) // Creating a new tensor:
                    {
                        int[] shp = call.input( 1 ).getNDConf().shape();
                        Tsr<Double> output = Tsr.of( shp, 0.0 ).getUnsafe().setIsIntermediate( true );
                        output.setIsVirtual( false );
                        try {
                            device.store( output );
                        } catch( Exception e ) {
                            e.printStackTrace();
                        }
                        call = call.withInputAt( 0, output );
                    }
                    return call;
                }
        )
        .buildFunAlgorithm();

        setAlgorithm(
            Activation.class,
            activation.setImplementationFor(
                CPU.class, new CPUElementwiseActivation(ScalarFun.IDENTITY)
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation
                    .compiler()
                    .arity( 3 )
                    .kernelSource( activation.getKernelSource() )
                    .activationSource( "output = input;" )
                    .differentiationSource( "output = 1;" )
                    .kernelPostfix( this.getIdentifier() )
                    .execution(
                        call -> {
                            int offset = ( call.input( Number.class, 0 ) != null ) ? 0 : 1;
                            int gwz =
                                    ( call.input( Number.class, 0 ) != null )
                                            ? call.input( Number.class, 0 ).size()
                                            : call.input( Number.class, 1 ).size();
                            call.getDevice().getKernel(call)
                                    .passAllOf( call.input( Number.class, offset ) )
                                    .passAllOf( call.input( Number.class, offset + 1 ) )
                                    .pass( call.input( Number.class, 0 ).rank() )
                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                    .call( gwz );

                            return call.input( 0 );
                        }
                    )
                    .build()
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
