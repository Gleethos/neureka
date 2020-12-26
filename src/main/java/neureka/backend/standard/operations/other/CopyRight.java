package neureka.backend.standard.operations.other;

import neureka.Tsr;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.Operation;
import neureka.backend.standard.algorithms.Activation;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.devices.opencl.OpenCLDevice;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.calculus.Function;
import neureka.backend.api.ExecutionCall;

import java.util.List;

public class CopyRight extends AbstractOperation {

    public CopyRight()
    {
        super("inject_right", ">", 2,true, false, false, true);

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get( i ) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" -> ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        DefaultOperatorCreator<TertiaryNDIConsumer> activationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                    else return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                };


        Activation activation = new Activation()
        .setBackwardADAnalyzer( call -> false )
        .setForwardADAnalyzer( call -> false )
        .setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
        )
        .setCallHook( (caller, call ) -> null )
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[1+offset], tsrs[offset]}, -1, Operation.instance("idy") );
                }
        )
        .build();

        setAlgorithm(Activation.class,
                activation.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call -> {
                                    int offset = ( call.getTensor( 0 ) == null ) ? 1 : 0;
                                    ExecutionCall<HostCPU> newCall = new ExecutionCall<>(
                                            call.getDevice(),
                                            new Tsr[]{call.getTensor(1+offset), call.getTensor(offset)},
                                            -1,
                                            call.getOperation()
                                    );
                                    Operation.instance("idy")
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
                                    int offset = ( call.getTensor( 0 ) == null ) ? 1 : 0;
                                    ExecutionCall<OpenCLDevice> newCall = new ExecutionCall<>(
                                            call.getDevice(),
                                            new Tsr[]{call.getTensor(1+offset), call.getTensor(offset)},
                                            -1,
                                            call.getOperation()
                                    );
                                    Operation.instance("idy")
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
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
    }
}
