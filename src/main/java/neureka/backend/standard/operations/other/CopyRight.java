package neureka.backend.standard.operations.other;

import neureka.Tsr;
import neureka.backend.standard.implementations.Activation;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.host.execution.HostExecutor;
import neureka.devices.opencl.OpenCLDevice;
import neureka.devices.opencl.execution.CLExecutor;
import neureka.calculus.Function;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperationType;
import neureka.backend.api.operations.OperationType;

import java.util.List;

public class CopyRight extends AbstractOperationType {

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
                getDefaultImplementation().supplyADAgentFor( f, call, forward )
        )
        .setCallHook( (caller, call ) -> null )
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[1+offset], tsrs[offset]}, -1, OperationType.instance("idy") );
                }
        )
        .build();

        setImplementation(Activation.class,
                activation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call -> {
                                    int offset = ( call.getTensor( 0 ) == null ) ? 1 : 0;
                                    ExecutionCall<HostCPU> newCall = new ExecutionCall<>(
                                            call.getDevice(),
                                            new Tsr[]{call.getTensor(1+offset), call.getTensor(offset)},
                                            -1,
                                            call.getOperation()
                                    );
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecutor(HostExecutor.class)
                                            .getExecution().run(call);
                                },
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = ( call.getTensor( 0 ) == null ) ? 1 : 0;
                                    ExecutionCall<OpenCLDevice> newCall = new ExecutionCall<>(
                                            call.getDevice(),
                                            new Tsr[]{call.getTensor(1+offset), call.getTensor(offset)},
                                            -1,
                                            call.getOperation()
                                    );
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecutor(CLExecutor.class)
                                            .getExecution().run(call);
                                },
                                3
                        )
                )
        );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
    }
}
