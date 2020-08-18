package neureka.calculus.environment.operations.other;

import neureka.Tsr;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.OpenCLDevice;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.Activation;
import neureka.calculus.environment.implementations.Convolution;

public class CopyRight extends OperationType {

    public CopyRight()
    {
        super("", ">", 2,true, false, false, false, false);

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" -> ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        DefaultOperatorCreator<TertiaryNDXConsumer> activationCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                    else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                };

        Activation activation = new Activation()
        .setADAnalyzer(
                call -> {
                    if ( call.getType().supports(Convolution.class) ) return false;
                    if ( call.getType().identifier().equals(",") ) return false; //Reshape
                    Tsr last = null;
                    for ( Tsr t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        ).setADAgentCreator(
             null
        ).setCallHock(
                (caller, call) -> null
        ).setRJAgent(
                ( call, goDeeperWith ) -> null
        ).setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[0] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[1+offset], tsrs[offset]}, -1, OperationType.instance("idy") );
                }
        );

        setImplementation(Activation.class,
                activation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call -> {
                                    int offset = ( call.getTensor(0) == null ) ? 1 : 0;
                                    ExecutionCall<HostCPU> newCall = new ExecutionCall<>(
                                            call.getDevice(),
                                            new Tsr[]{call.getTensor(1+offset), call.getTensor(offset)},
                                            -1,
                                            call.getType()
                                    );
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecutor(HostExecutor.class)
                                            .getExecution().call(call);
                                },
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = ( call.getTensor(0) == null ) ? 1 : 0;
                                    ExecutionCall<OpenCLDevice> newCall = new ExecutionCall<>(
                                            call.getDevice(),
                                            new Tsr[]{call.getTensor(1+offset), call.getTensor(offset)},
                                            -1,
                                            call.getType()
                                    );
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecutor(CLExecutor.class)
                                            .getExecution().call(call);
                                },
                                3
                        )
                )
        );
    }

}
