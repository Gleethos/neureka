package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;

public class CopyRight extends AbstractOperation {

    public CopyRight()
    {
        super(
                new OperationBuilder()
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
        .setHandleInsteadOfDevice( AbstractOperation::executeMe )
        .setHandleRecursivelyAccordingToArity( (call, goDeeperWith ) -> null )
        .setInstantiateNewTensorsForExecutionIn(
                call -> {
                    Tsr<?>[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    //tsrs[0] = tsrs[2];
                    return
                            ExecutionCall.of(tsrs[1+offset], tsrs[offset])
                                            .andArgs( Arg.DerivIdx.of( -1 ) )
                                            .running( Neureka.get().context().getOperation("idy") ) // This routes to another operation!
                                            .on( call.getDevice() );
                }
        )
        .build();

        setAlgorithm(Activation.class,
                activation.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call -> {
                                    int offset = 1;
                                    Tsr[] args = { call.getTsrOfType( Number.class, 1+offset), call.getTsrOfType( Number.class, offset)};
                                    ExecutionCall<HostCPU> newCall =
                                            ExecutionCall.of(args)
                                                            .andArgs(Arg.DerivIdx.of(-1))
                                                            .running(call.getOperation())
                                                            .on( call.getDevice() )
                                                            .forDeviceType(HostCPU.class);
                                    Neureka.get().context().getOperation("idy")
                                            .getAlgorithm(Activation.class)
                                            .getImplementationFor( HostCPU.class )
                                            .run(newCall);
                                    call.getTensors()[0] = args[1];
                                },
                                2
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        call -> {
                            int offset = 1;
                            Tsr[] args = { call.getTsrOfType( Number.class, 1+offset), call.getTsrOfType( Number.class, offset)};
                            ExecutionCall<OpenCLDevice> newCall = ExecutionCall.of(args)
                                                                                .andArgs(Arg.DerivIdx.of( -1 ))
                                                                                .running(call.getOperation())
                                                                                .on( call.getDevice() )
                                                                                .forDeviceType(OpenCLDevice.class);
                            Neureka.get().context().getOperation("idy")
                                    .getAlgorithm(Activation.class)
                                    .getImplementationFor( OpenCLDevice.class )
                                    .run(newCall);
                            call.getTensors()[0] = args[1];
                        }
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
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
    }
}
