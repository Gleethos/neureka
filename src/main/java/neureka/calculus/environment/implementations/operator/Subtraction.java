package neureka.calculus.environment.implementations.operator;

import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Subtraction extends OperationType {

    private static final DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
            (inputs, d) -> {
                double[] t1_val = inputs[1].value64();
                double[] t2_val = inputs[2].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] - t2_val[inputs[2].i_of_idx(t2Idx)];
                } else return (t0Idx, t1Idx, t2Idx) -> (d == 0) ? 1.0 : -1.0;
            };

    public Subtraction(){

        super(
                "subtract", "-", -1, true, false, false, false, false
        );

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<PrimaryNDXConsumer> operationCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if (d < 0) {
                        return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] - t2_val[inputs[2].i_of_idx(t1Idx)];
                    } else return t1Idx -> (d == 0) ? 1.0 : -1.0;
                };

        Operation operation = new Operation();

        setImplementation(Operation.class,
                operation.setExecutor(
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
                                                                        operationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setExecutor(
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
                                "output = input1 - input2;  \n",
                                "if(d==0){                 \n" +//drn and src2 switch:
                                        "    output = 1;              \n" +
                                        "} else {                     \n" +
                                        "    output = -1;               " +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarOperatorCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] - value;
                    else {
                        if (d == 0) return t1Idx -> 1; else return t1Idx -> -1;
                    }
                };

        Scalarization scalarization = new Scalarization();

        setImplementation(Scalarization.class,
                scalarization.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call -> {
                                    int offset = (call.getTensor(2).isVirtual() || call.getTensor(2).size() == 1) ? 1 : 0;
                                    double value = call.getTensor(1+offset).value64(0);
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
                ).setExecutor(
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
                                "output = input1 - value;\n",
                                "if(d==0){     \n" +//drn and src2 switch:
                                        "    output = 1;  \n" +
                                        "} else {         \n" +
                                        "    output = -1;   " +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //________________
        // BROADCASTING :

        setImplementation(Broadcast.class,
                new Broadcast() // add _creator
        );

        //______________________
        // RELATED OPERATIONS :

        new OperationType(
                "", ((char) 171) + "-", 3, true, false, false, false, false
        );
        new OperationType(
                "", "-" + ((char) 187), 3, true, false, false, false, false
        );

        // Convolution:


        new OperationType(
                "", "s", 2, true, false, true, false, false
        );
        new OperationType(
                "", ((char) 171) + "s", 3, true, false, true, false, false
        );
        new OperationType(
                "", "s" + ((char) 187), 3, true, false, true, false, false
        );


    }


}
