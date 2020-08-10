package neureka.calculus.environment.operations.function;

import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.*;

public class Tanh extends OperationType
{

    private DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
            ( inputs, d ) ->
            {
                double[] t1_val = inputs[1].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                        return input / Math.pow(1 + Math.pow(input, 2), 0.5);
                    };
                } else {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                        return 1 - Math.pow(input / Math.pow(1 + Math.pow(input, 2), 0.5), 2);
                    };
                }
            };

    public Tanh()
    {
        super (
                "tanh",
                "tanh",
                1,
                false,
                false,
                false,
                true,
                true
        );
        Activation typeImplementation = new Activation();

        setImplementation(
                Activation.class,
                typeImplementation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
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
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
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
                                "output = input/pow(1+pow(input, 2.0f), 0.5f);\n",
                                "output = 1-pow(input/pow((1.0f+pow(input,2.0f)),0.5f), 2.0f);\n",
                                this // OperationType
                        )
                )
        );




    }





}

