package neureka.calculus.environment.operations.function;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.*;
import neureka.calculus.factory.assembly.FunctionBuilder;

public class Cosinus extends OperationType {

    private DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
            (inputs, d)->{
                double[] t1_val = inputs[1].value64();
                if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.cos(t1_val[inputs[1].i_of_idx(t1Idx)]);
                else return (t0Idx, t1Idx, t2Idx) -> -Math.sin(t1_val[inputs[1].i_of_idx(t1Idx)]);
            };

    public Cosinus()
    {
        super (
                "cosinus",
                "cos" ,
                1,
                false,
                false,
                false,
                true,
                true
        );

        setStringifier(
                children -> {
                    String expression = String.join( ", ", children );
                    if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                        return "cos" + expression;
                    }
                    return "cos" + "(" + expression + ")";
                }
        );

        Activation typeImplementation = new Activation()
        .setADAnalyzer(
                call -> {
                    Tsr last = null;
                    for ( Tsr t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        ).setADAgentCreator(
    (Function f, Tsr derivv, ExecutionCall<Device> call, boolean forward ) ->
    {
        Function mul = Function.Detached.MUL;
        if (
            derivv != null
        ) {
            return new ADAgent(
                    () -> derivv
                ).withForward(
                    ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, derivv})
                ).withBackward(
                    null
                );
        }
        Tsr[] inputs = call.getTensors();
        int d = call.getDerivativeIndex();
        if( forward )
        {
            Tsr deriv = f.derive(inputs, d);
            return new ADAgent(
                    () -> deriv
                ).withForward(
                    ( t, derivative ) -> mul.call(new Tsr[]{derivative, deriv})
                ).withBackward(
                    null
                );
        }
        else
        {

            {
                Tsr deriv = f.derive(inputs, d);
                return new ADAgent(
                            ()->deriv
).withForward(
                            (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
).withBackward(
                            (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv})
);
            }
        }
    }
        ).setCallHock(
                ( caller, call ) -> null
        ).setRJAgent(
                ( call, goDeeperWith ) -> null
        ).setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[1].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual(false);
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
        );

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
                                "output = cos(input);\n", // activationSource
                                "output = -sin(input);\n", //differentiationSource
                                this // OperationType
                        )
                )
        );

    }



}
