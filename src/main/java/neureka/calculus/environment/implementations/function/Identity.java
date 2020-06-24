package neureka.calculus.environment.implementations.function;

import neureka.Tsr;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.host.execution.HostExecution;
import neureka.acceleration.opencl.OpenCLDevice;
import neureka.acceleration.opencl.execution.CLExecution;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.Type;
import neureka.calculus.environment.executors.*;

public class Identity extends OperationType {

    private Type.OperatorCreator _creator =
            (inputs, d) -> {
                double[] t1_val = inputs[1].value64();
                if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
            };

    public Identity(){

        super("identity", "idy" , 1, false, false, false, true, true);

        Activation typeImplementation =
                new Activation(
                        "output = input;\n",
                        "output = input;\n",
                        _creator
                        );
        setImplementation(
                Activation.class,
                typeImplementation.setExecution (
                        HostCPU.class,
                        new HostExecution(
                                call  ->
                                        call.getDevice().getExecutor()
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
                                "output = input;\n", // activationSource
                                "output = input;\n", //differentiationSource
                                this // OperationType
                        )
                )
        );

        Scalarization scalarization =
                new Scalarization(
                        "output = value;\n",
                        "output = value;\n",
                        null
                );
        setImplementation(Scalarization.class,
                scalarization.setExecution (
                        HostCPU.class,
                        new HostExecution(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        (start, end) ->
                                                                Activation.activate(
                                                                        call.getTensor(0), start, end,
                                                                        null
                                                                        //_creator.create(
                                                                        //        tsrs, scalar, d
                                                                        //)
                                                                )
                                                ),
                                3
                        )
                ).setExecution(
                        OpenCLDevice.class,
                        new CLExecution(
                                call -> {
                                    Tsr t = call.getTensor(0);
                                    int gwz = t.size();
                                    call.getDevice().getKernel(call)
                                            .pass(t)
                                            .pass(t)
                                            .pass((float)call.getTensor(1).value64(0))
                                            .pass(t.rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                typeImplementation.getKernelSource(), // kernelSource
                                "output = input;\n", // activationSource
                                "output = input;\n", //differentiationSource
                                this // OperationType
                        )
                )
        );


    }

}
