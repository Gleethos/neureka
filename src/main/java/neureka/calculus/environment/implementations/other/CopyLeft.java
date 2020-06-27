package neureka.calculus.environment.implementations.other;

import neureka.Tsr;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.OpenCLDevice;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;
import neureka.calculus.environment.executors.Activation;

public class CopyLeft extends OperationType {

    public CopyLeft(){

        super(
                "", "<", 2,true, false, false, false, false
        );

        DefaultOperatorCreator<TertiaryNDXConsumer> activationCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                    else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                };

        Activation activation = new Activation();

        setImplementation(Activation.class,
                activation.setExecution (
                        HostExecutor.class,
                        new HostExecutor(
                                call -> {
                                    int offset = ( call.getTensor(0) == null ) ? 1 : 0;
                                    OperationTypeImplementation.ExecutionCall<HostCPU> newCall = new OperationTypeImplementation.ExecutionCall<>(
                                            call.getDevice(),
                                            new Tsr[]{call.getTensor(offset), call.getTensor(1+offset)},
                                            -1,
                                            call.getType()
                                    );
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecution(HostExecutor.class)
                                            .getExecution().call(call);
                                },
                                3
                        )
                ).setExecution(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = ( call.getTensor(0) == null ) ? 1 : 0;
                                    OperationTypeImplementation.ExecutionCall<OpenCLDevice> newCall = new OperationTypeImplementation.ExecutionCall<>(
                                            call.getDevice(),
                                            new Tsr[]{call.getTensor(offset), call.getTensor(1+offset)},
                                            -1,
                                            call.getType()
                                    );
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecution(CLExecutor.class)
                                            .getExecution().call(call);
                                },
                                3
                        )
                )
        );
    }



}
