package neureka.calculus.environment.implementations.indexer;

import neureka.acceleration.host.HostCPU;
import neureka.acceleration.host.execution.HostExecution;
import neureka.acceleration.opencl.OpenCLDevice;
import neureka.acceleration.opencl.execution.CLExecution;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.Type;
import neureka.calculus.environment.executors.*;

public class Product extends OperationType {

    public Product()
    {
        super (
                "product",
                "prod",
                1,
                false,
                true,
                false,
                true,
                true
        );
        setImplementation(Activation.class,
                new Activation(
                        "output = input;",
                        "output = 1;",
                        null
                )
        );
        Type.OperatorCreator _creator =
                (inputs, d) ->
                {
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

        Broadcast typeImplementation =
                new Broadcast(
                        "value = src1 * src2;\n",
                        "value += handle * drain;\n",
                        _creator
                );


        setImplementation (
                Broadcast.class,
                typeImplementation.setExecution (
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
                                typeImplementation.getKernelSource(), // kernelSource
                                "output = input/pow(1+pow(input, 2.0f), 0.5f);\n",
                                "output = 1-pow(input/pow((1.0f+pow(input,2.0f)),0.5f), 2.0f);\n",
                                this // OperationType
                        )
                )
        );

    }




}
