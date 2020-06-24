package neureka.calculus.environment.implementations.operator;

import neureka.acceleration.host.HostCPU;
import neureka.acceleration.host.execution.HostExecution;
import neureka.acceleration.opencl.OpenCLDevice;
import neureka.acceleration.opencl.execution.CLExecution;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Addition extends OperationType {

    private static final OperatorCreator _creator =
            (inputs, d) -> {
                double[] t1_val = inputs[1].value64();
                double[] t2_val = inputs[2].value64();
                if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + t2_val[inputs[2].i_of_idx(t2Idx)];
                else return (t0Idx, t1Idx, t2Idx) -> 1.0;
            };

    private static final Broadcast _broadcast = new Broadcast(
            "value = src1 + src2;\n",
            "value += 1 * drain;\n",
            _creator
    );

    public Addition()
    {
        super (
                "add",
                "+",
                -1,
                true,
                false,
                false,
                true,
                false
        );
        setImplementation(Broadcast.class,
                _broadcast
                .setExecution (
                        HostCPU.class,
                        new HostExecution(
                                ( device, call ) ->
                                        device.getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor(0),
                                                                        start, end,
                                                                        _creator.create(call.getTensors(), -1)
                                                                )
                                                ),
                                3
                        )
                ).setExecution(
                        OpenCLDevice.class,
                        new CLExecution(
                                ( device, call ) -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    device.getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                _broadcast.getKernelSource(), // kernelSource
                                "value = src1 + src2;\n",
                                "value += 1 * drain;\n",
                                this // OperationType
                        )
                )
        );

        //__________________
        // IMPLEMENTATION :

        Operation _operation = new Operation(
                "output = input1 + input2;\n",
                "output = 1;\n",
                _creator
        );

        setImplementation(Operation.class,
                _operation
                        .setExecution (
                                HostCPU.class,
                                new HostExecution(
                                        ( device, call ) ->
                                                device.getExecutor()
                                                        .threaded (
                                                                call.getTensor(0).size(),
                                                                ( start, end ) ->
                                                                        Broadcast.broadcast (
                                                                                call.getTensor(0),
                                                                                call.getTensor(1),
                                                                                call.getTensor(2),
                                                                                call.getDerivativeIndex(),
                                                                                start, end,
                                                                                _creator.create(call.getTensors(), -1)
                                                                        )
                                                        ),
                                        3
                                )
                        ).setExecution(
                        OpenCLDevice.class,
                        new CLExecution(
                                ( device, call ) -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    device.getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                _broadcast.getKernelSource(), // kernelSource
                                "value = src1 + src2;\n",
                                "value += 1 * drain;\n",
                                this // OperationType
                        )
                )
        );


        setImplementation(Scalarization.class,
                new Scalarization(
                        "output = input1 + value;\n",
                        "output = 1;\n",
                        (inputs, value, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + value;
                            else return (t0Idx, t1Idx, t2Idx) -> 1;
                        })
        );

        new OperationType(
                "", ((char) 171) + "+", 3, true, false, false, false, false
        ).setImplementation(Broadcast.class, _broadcast);

        new OperationType(
                "", "+" + ((char) 187), 3, true, false, false, false, false
        ).setImplementation(Broadcast.class, _broadcast);

        // Convolutoion:

        new OperationType(
                "add", "a", 2, true, false, true, false, false
        ).setImplementation(Convolution.class,
                new Convolution(
                        "value = src1 + src2;\n",
                        "value += 1 * drain;\n",
                        null
                )
        );

        new OperationType(
                "", ((char) 171) + "a", 3, true, false, true, false, false
        );
        new OperationType(
                "", "a" + ((char) 187), 3, true, false, true, false, false
        );


    }

}
