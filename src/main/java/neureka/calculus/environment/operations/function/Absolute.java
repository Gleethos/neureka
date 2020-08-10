package neureka.calculus.environment.operations.function;

import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.*;

public class Absolute extends OperationType {

    private DefaultOperatorCreator<TertiaryNDXConsumer> _activationCreator =
    (inputs, d)->{
        double[] t1_val = inputs[1].value64();
        if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.abs(t1_val[inputs[1].i_of_idx(t1Idx)]);
        else return (t0Idx, t1Idx, t2Idx) -> (t1_val[inputs[1].i_of_idx(t1Idx)] < 0) ? -1 : 1;
    };

    public Absolute()
    {
        super("absolute", "abs" , 1, false, false, false, true, true);

        Activation typeImplementation = new Activation();

        setImplementation(
                Activation.class,
                typeImplementation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call  ->
                                        call.getDevice().getExecutor()
                                        .threaded(
                                            call.getTensor(0).size(),
                                            ( start, end ) ->
                                                    Activation.activate(
                                                            call.getTensor(0),
                                                            start, end,
                                                            _activationCreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                    call.getDevice().getKernel( call )
                                            .pass( call.getTensor( offset ) )
                                            .pass( call.getTensor(offset + 1 ) )
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                typeImplementation.getKernelSource(), // kernelSource
                                "output = fabs(input);\n", // activationSource
                                "output = (input < 0) ? -1 : 1;\n", //differentiationSource
                                this // OperationType
                        )
                )
        );
    }

}
