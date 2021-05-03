package neureka.backend.standard.operations.other;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.api.operations.OperationFactory;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;

public class CopyRight extends AbstractOperation {

    public CopyRight()
    {
        super(
                new OperationFactory()
                        .setFunction(         "inject_right"    )
                        .setOperator(         ">"        )
                        .setArity(            2          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( false       )
                        .setIsInline(         true      )
        );

        DefaultOperatorCreator<TertiaryNDIConsumer> activationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                    else return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                };


        Activation activation = new Activation()
        .setCanPerformBackwardADFor( call -> false )
        .setCanPerformForwardADFor( call -> false )
        .setSupplyADAgentFor(
            ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
        )
        .setHandleInsteadOfDevice( (caller, call ) -> null )
        .setHandleRecursivelyAccordingToArity( (call, goDeeperWith ) -> null )
        .setInstantiateNewTensorsForExecutionIn(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return
                            ExecutionCall.builder()
                                .device( call.getDevice() )
                                .tensors( new Tsr[]{tsrs[1+offset], tsrs[offset]} )
                                .derivativeIndex( -1 )
                                .operation( OperationContext.get().instance("idy") )
                                .build();
                }
        )
        .build();

        setAlgorithm(Activation.class,
                activation.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call -> {
                                    int offset = ( call.getTsrOfType( Number.class, 0 ) == null ) ? 1 : 0;
                                    ExecutionCall<HostCPU> newCall =
                                            ExecutionCall.builder()
                                                .device( call.getDevice() )
                                                .tensors( new Tsr[]{call.getTsrOfType( Number.class, 1+offset), call.getTsrOfType( Number.class, offset)} )
                                                .derivativeIndex( -1 )
                                                .operation( call.getOperation() )
                                                .build()
                                                .forDeviceType(HostCPU.class);
                                    OperationContext.get().instance("idy")
                                            .getAlgorithm(Activation.class)
                                            .getImplementationFor( HostCPU.class )
                                            .run(call);
                                },
                                2
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = ( call.getTsrOfType( Number.class, 0 ) == null ) ? 1 : 0;
                                    ExecutionCall<OpenCLDevice> newCall = ExecutionCall.builder()
                                            .device( call.getDevice() )
                                            .tensors( new Tsr[]{call.getTsrOfType( Number.class, 1+offset), call.getTsrOfType( Number.class, offset)} )
                                            .derivativeIndex( -1 )
                                            .operation( call.getOperation() )
                                            .build()
                                            .forDeviceType(OpenCLDevice.class);
                                    OperationContext.get().instance("idy")
                                            .getAlgorithm(Activation.class)
                                            .getImplementationFor( OpenCLDevice.class )
                                            .run(call);
                                },
                                2
                        )
                )
        );
    }

    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" -> ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int d ) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
    }
}
