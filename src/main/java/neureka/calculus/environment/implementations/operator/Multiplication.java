package neureka.calculus.environment.implementations.operator;

import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Multiplication extends OperationType {


    private static final DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
            (inputs, d) -> {
                double[] t1_val = inputs[1].value64();
                double[] t2_val = inputs[2].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                } else {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        if (d == 0) return t2_val[inputs[2].i_of_idx(t2Idx)];
                        else return t1_val[inputs[1].i_of_idx(t1Idx)];
                    };
                }
            };

    public Multiplication()
    {
        super(
                "multiply", "*", -1,
                true, false, false, true, false
        );

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<PrimaryNDXConsumer> defaultOperatorcreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t1Idx)];
                    } else {
                        return t1Idx -> {
                            if ( d == 0 ) return t2_val[inputs[2].i_of_idx(t1Idx)];
                            else return t1_val[inputs[1].i_of_idx(t1Idx)];
                        };
                    }
                };

        Operation operation =
                new Operation(
                        _creator
                );

        setImplementation(Operation.class,
                operation.setExecution (
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Operation.operate (
                                                                        call.getTensor(0),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        defaultOperatorcreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                "output = input1 * input2;\n",
                                "if(d==0){output = input2;}else{output = input1;}\n",
                                this // OperationType
                        )
                )
        );


        //________________
        // BROADCASTING :

        Broadcast broadcast =
                new Broadcast(
                        _creator
                );

        setImplementation(Broadcast.class,
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
                            "value = src1 * src2;\n",
                            "value += handle * drain;\n",
                            this // OperationType
                    )
            )
        );




        //___________________________
        // TENSOR SCALAR OPERATION :

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarOperatorCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if ( d < 0 ) return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] * value;
                    else {
                        if ( d == 0 ) return t1Idx -> value;
                        else return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)];
                    }
                };

        Scalarization scalarization =
                new Scalarization(
                        scalarOperatorCreator
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
                                                                    scalarOperatorCreator.create(call.getTensors(), value, -1)
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
                                "output = input1 * value;\n",
                                "if(d==0){output = value;}else{output = input1;}\n",
                                this // OperationType
                        )
                )
        );




        //__________________________
        // RELATED OPERATION TYPES :

        DefaultOperatorCreator<TertiaryNDXConsumer> xCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                };

        Broadcast xBroadcast =
                new Broadcast(
                        xCreator
                );

        new OperationType(
                "", ((char) 171) + "*", 3, true, false, false, false, false
        ).setImplementation(
                Broadcast.class,
                xBroadcast.setExecution (
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
                                                                        xCreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                xBroadcast.getKernelSource(), // kernelSource
                                "value = src1 * src2;\n",
                                "value += handle * drain;\n",
                                this // OperationType
                        )
                )
        );

        xBroadcast =
                new Broadcast(
                        xCreator
                );

        new OperationType(
                "", "*" + ((char) 187), 3, true, false, false, false, false
        ).setImplementation(
                Broadcast.class,
                xBroadcast.setExecution (
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
                                                                        xCreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                xBroadcast.getKernelSource(), // kernelSource
                                "value = src1 * src2;\n",
                                "value += handle * drain;\n",
                                this // OperationType
                        )
                )
        );





        // Convolution:

        Convolution convolution =
                new Convolution(
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            double[] t2_val = inputs[2].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if (d == 0) return t2_val[inputs[2].i_of_idx(t2Idx)];
                                    else return t1_val[inputs[1].i_of_idx(t1Idx)];
                                };
                            }
                        }
                );

        new OperationType(
                "multiply", "x", 2, true, false, true, false, false
        ).setImplementation(
                Convolution.class,
                convolution
                        .setExecution (
                        HostExecutor.class,
                                new HostExecutor(
                                        call ->
                                                call.getDevice().getExecutor()
                                                        .threaded (
                                                                call.getTensor(0).size(),
                                                                ( start, end ) ->
                                                                        Convolution.convolve (
                                                                                call.getTensor(0), call.getTensor(1), call.getTensor(2),
                                                                                call.getDerivativeIndex(), start, end,
                                                                                _creator.create(
                                                                                        call.getTensors(),
                                                                                        -1//call.getDerivativeIndex()
                                                                                )
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
                                            .pass(call.getDerivativeIndex())//call.getDerivativeIndex()
                                            .call(gwz);
                                },
                                3,
                                convolution.getKernelSource(), // kernelSource
                                "value = src1 * src2;\n",
                                "value += handle * drain;\n",
                                this // OperationType
                        )
                )
        );
        new OperationType(
                "inv_convolve_mul_left", ((char) 171) + "x", 3, true, false, true, false, false
        ).setImplementation(Convolution.class, convolution);
        new OperationType(
                "inv_convolve_mul_right", "x" + ((char) 187), 3, true, false, true, false, false
        ).setImplementation(Convolution.class, convolution);



    }





}
