package neureka.calculus.environment.implementations.operator;

import neureka.acceleration.host.HostCPU;
import neureka.acceleration.host.execution.HostExecution;
import neureka.acceleration.opencl.OpenCLDevice;
import neureka.acceleration.opencl.execution.CLExecution;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Power extends OperationType
{

    private final static DefaultOperatorCreator<TertiaryNDXConsumer> _creator = (inputs, d)->
    {
        double[] t1_val = inputs[1].value64();
        double[] t2_val = inputs[2].value64();
        if (d < 0) {
            return (t0Idx, t1Idx, t2Idx) -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], t2_val[inputs[2].i_of_idx(t2Idx)]);
        } else {
            return (t0Idx, t1Idx, t2Idx) -> {
                if (d == 0) {
                    return t2_val[inputs[2].i_of_idx(t2Idx)]
                            * Math.pow(
                            t1_val[inputs[1].i_of_idx(t1Idx)],
                            t2_val[inputs[2].i_of_idx(t2Idx)] - 1
                    );
                } else {
                    return Math.pow(
                            t1_val[inputs[1].i_of_idx(t1Idx)],
                            t2_val[inputs[2].i_of_idx(t2Idx)]
                    ) * Math.log(t1_val[inputs[1].i_of_idx(t1Idx)]);
                }
            };
        }
    };

    public Power()
    {
        super("power", "^", -1, true, false, false, false, false);

        //_____________________
        // DEFAULT OPERATION :

        setImplementation(Operation.class,
                new Operation(
                        "output = pow(input1, input2);",
                        "if(d==0) {                                    \n" +
                                "    output = input2 * pow(input1, input2-1.0f);  \n" +
                                "} else {                                         \n" +
                                "    output = pow(input1, input2) * log(input1);  \n" +
                                "}",
                        _creator
                )
        );

        //________________
        // BROADCASTING :

        Broadcast broadcast =
                new Broadcast(
                        "value += pow(src1, src2);",
                        "if(d==0){\n" +
                                "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                                "} else {\n" +
                                "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                                "}",
                        _creator
                );

        setImplementation(Broadcast.class,
                broadcast.setExecution (
                        HostCPU.class,
                        new HostExecution(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor(0), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        _creator.create(call.getTensors(), -1)
                                                                )
                                                ),
                                3
                        )
                ).setExecution(
                        OpenCLDevice.class,
                        new CLExecution(
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
                                "value += pow(src1, src2);",
                                "if(d==0){\n" +
                                        "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                                        "} else {\n" +
                                        "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarCreator =
                ( inputs, value, d )->{
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) {
                        return t1Idx -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value);
                    } else {
                        if(d==0){
                            return t1Idx -> value*Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value-1);
                        } else {
                            return t1Idx -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value)*Math.log(value);
                        }
                    }
                };


        Scalarization scalarization =
                new Scalarization(
                        "output = pow(input1, value);",
                        "if ( d==0 ) {                                     \n" +
                                "    output = value * pow(input1, value-(float)1 );   \n" +
                                "} else {                                             \n" +
                                "    output = pow(input1, value) * log(value);        \n" +
                                "}",
                            scalarCreator
                        );

        setImplementation(
                Scalarization.class,
                scalarization.setExecution (
                        HostCPU.class,
                        new HostExecution(
                                call -> {
                                    double value = call.getTensor(0).value64(2);
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTensor(0).size(),
                                                    ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTensor(0),
                                                                    start, end,
                                                                    scalarCreator.create(call.getTensors(), value, -1)
                                                            )
                                            );
                                },
                                3
                        )
                ).setExecution(
                        OpenCLDevice.class,
                        new CLExecution(
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
                                "output = pow(input1, value);",
                                "if ( d==0 ) {                                     \n" +
                                        "    output = value * pow(input1, value-(float)1 );   \n" +
                                        "} else {                                             \n" +
                                        "    output = pow(input1, value) * log(value);        \n" +
                                        "}",
                                this // OperationType
                        )
                )
        );




        //__________________________
        // RELATED OPERATION TYPES :

        new OperationType("inv_power_left", ((char)171)+"^", 3, true, false, false, false, false);
        new OperationType("inv_power_right", "^" + ((char) 187), 3, true, false, false, false, false);

        // Convolution:

        new OperationType(
                "power", "p", 2, true, false, true, false, false
        ).setImplementation(Convolution.class,
                new Convolution(
                        "value += pow(src1, src2);",
                        "if(d==0){\n" +
                                "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                                "} else {\n" +
                                "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                                "}",
                        null
                )
        );
        new OperationType("", ((char) 171) + "p", 3, true, false, true, false, false);
        new OperationType("", "p" + ((char) 187), 3, true, false, true, false, false);




    }

}
