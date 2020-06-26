package neureka.calculus.environment.implementations.function;

import neureka.acceleration.host.HostCPU;
import neureka.acceleration.host.execution.HostExecution;
import neureka.acceleration.opencl.OpenCLDevice;
import neureka.acceleration.opencl.execution.CLExecution;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Gaussian extends OperationType {

    private DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
            (inputs, d)->{
                double[] t1_val = inputs[1].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> Math.pow(Math.E, -Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], 2));
                } else {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                        return -2 * input * Math.pow(Math.E, -Math.pow(input, 2));
                    };

                }
            };

    public Gaussian(){

        super("gaussian", "gaus", 1, false, false, false, true, true);

        Activation typeImplementation =
                new Activation(
                        "output =\n" +
                                "    (float)pow(\n" +
                                "        (float)M_E,\n" +
                                "        -(float)pow(\n" +
                                "            (float)input,\n" +
                                "            (float)2\n" +
                                "        )\n" +
                                "    );\n",
                        "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                        _creator
                );

        setImplementation(
                Activation.class,
                typeImplementation.setExecution (
                        HostExecution.class,
                        new HostExecution(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor(0),
                                                                        start, end,
                                                                        _creator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setExecution(
                        CLExecution.class,
                        new CLExecution(
                                call -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                typeImplementation.getKernelSource(), // kernelSource
                                "output =\n" +
                                        "    (float)pow(\n" +
                                        "        (float)M_E,\n" +
                                        "        -(float)pow(\n" +
                                        "            (float)input,\n" +
                                        "            (float)2\n" +
                                        "        )\n" +
                                        "    );\n",
                                "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                                this // OperationType
                        )
                )
        );

    }

}
