package neureka.calculus.backend.operations.other;

import neureka.Tsr;
import neureka.device.Device;
import neureka.device.host.HostCPU;
import neureka.device.host.execution.HostExecutor;
import neureka.device.opencl.OpenCLDevice;
import neureka.device.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.implementations.functional.Activation;

import java.util.List;

public class CopyRight extends AbstractOperationType {

    public CopyRight()
    {
        super("inject_right", ">", 2,true, false, false, true);

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
        .setBackwardADAnalyzer( call -> false )
        .setForwardADAnalyzer( call -> false )
        .setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
            {
                Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                Function mul = Function.Detached.MUL;
                if (
                    ctxDerivative != null
                ) {
                    return new ADAgent(
                            ctxDerivative
                        ).withForward(
                            ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative})
                        ).withBackward(
                            null
                        );
                }
                Tsr[] inputs = call.getTensors();
                int d = call.getDerivativeIndex();
                if( forward )
                {
                    Tsr deriv = f.derive(inputs, d);
                    return new ADAgent(
                            deriv
                        ).withForward(
                            ( t, derivative ) -> mul.call(new Tsr[]{derivative, deriv})
                        ).withBackward(
                            null
                        );
                }
                else
                {

                    Tsr deriv = f.derive(inputs, d);
                    return new ADAgent(
                                    deriv
                                ).withForward(
                                    (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
                                ).withBackward(
                                    (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv})
                                );
                }
            }
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
                                            .getExecution().run(call);
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
                                            .getExecution().run(call);
                                },
                                3
                        )
                )
        );
    }

    @Override
    public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
    }
}
