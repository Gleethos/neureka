package neureka.calculus.environment.implementations.operator;

import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;


public class Division extends OperationType
{
    private static final DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
    (inputs, d) -> {
        double[] t1_val = inputs[1].value64();
        double[] t2_val = inputs[2].value64();
        if (d < 0) {
            return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] / t2_val[inputs[2].i_of_idx(t2Idx)];
        } else {
            return (t0Idx, t1Idx, t2Idx) -> {
                if (d == 0) {
                    return 1 / t2_val[inputs[2].i_of_idx(t2Idx)];
                } else {
                    return -(t1_val[inputs[1].i_of_idx(t1Idx)] / Math.pow(t2_val[inputs[2].i_of_idx(t2Idx)], 2));
                }
            };
        }
    };

    public Division()
    {

        super(
                "divide", "/", -1,
                true,
                false,
                false,
                false,
                false
        );


        //_____________________
        // DEFAULT OPERATION :

        Operation operation =
                new Operation(
                        "output = input1 / input2;\n",
                        "if(d==0){\n" +
                                "    output = 1/input2;\n" +
                                "} else {\n" +
                                "    output = -input2 /(float)pow(input1, 2.0f);\n" +
                                "}",
                        _creator
                );

        setImplementation(
                Operation.class, operation.setExecution (
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor(0),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        _creator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setExecution(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                operation.getKernelSource(), // kernelSource
                                "output = input1 / input2;\n",
                                "if(d==0){\n" +
                                        "    output = 1/input2;\n" +
                                        "} else {\n" +
                                        "    output = -input2 /(float)pow(input1, 2.0f);\n" +
                                        "}",
                                this // OperationType
                        )
                )
        );


        //________________
        // BROADCASTING :

        Broadcast broadcast =
                new Broadcast(
                        "value = src1 / src2;\n",
                        "if(d==0){\n" +
                                "    value += (1/handle) * drain;\n" +
                                "} else {\n" +
                                "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                                "}",
                        _creator
                );
        setImplementation(
                Broadcast.class,
                broadcast.setExecution (
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor(0), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        _creator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setExecution(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                broadcast.getKernelSource(), // kernelSource
                                "value = src1 / src2;\n",
                                "if(d==0){\n" +
                                        "    value += (1/handle) * drain;\n" +
                                        "} else {\n" +
                                        "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) {
                        return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] / value;
                    } else {
                        if (d == 0) return t1Idx -> 1 / value;
                        else return t1Idx -> -value / Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], 2);
                    }
                };

        Scalarization scalarization = new Scalarization(
                "output = input1 / value;\n",
                "if(d==0){\n" +
                        "    output = 1/value;\n" +
                        "} else {\n" +
                        "    output = -value /(float)pow(input1, 2.0f);\n" +
                        "}",
                scalarCreator
        );


        setImplementation(
                Scalarization.class,
                scalarization.setExecution (
                        HostExecutor.class,
                        new HostExecutor(
                                call -> {
                                    double value = call.getTensor(0).value64(2);
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTensor(0).size(),
                                                    ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTensor(0),
                                                                    start, end,
                                                                    scalarCreator.create(call.getTensors(), value, call.getDerivativeIndex())
                                                            )
                                            );
                                },
                                3
                        )
                ).setExecution(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(2).isVirtual() || call.getTensor(2).size() == 1)?1:0;
                                    int gwz = call.getTensor(0).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(0))
                                            .pass(call.getTensor(0))
                                            .pass((float)call.getTensor(1+offset).value64(0))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                scalarization.getKernelSource(), // kernelSource
                                "output = input1 / value;\n",
                                "if(d==0){\n" +
                                        "    output = 1/value;\n" +
                                        "} else {\n" +
                                        "    output = -value /(float)pow(input1, 2.0f);\n" +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //__________________________
        // RELATED OPERATION TYPES :


        new OperationType(
                "inv_division_left", ((char) 171) + "/", 3, true, false, false, false, false
        );
        new OperationType(
                "inv_division_right", "/" + ((char) 187), 3, true, false, false, false, false
        );

        // Convolution:

        new OperationType(
                "divide", "d", 2, true, false, true, false, false
        ).setImplementation(
                Convolution.class,
                new Convolution(
                        "value = src1 / src2;\n",
                        "if(d==0) {\n" +
                                "    value += (1/handle) * drain;\n" +
                                "} else {\n" +
                                "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                                "}",
                        null
                )
        );

        new OperationType(
                "", ((char) 171) + "d", 3, true, false, true, false, false
        );
        new OperationType(
                "", "d" + ((char) 187), 3, true, false, true, false, false
        );

    }



}
