package neureka.calculus.environment.operations.indexer;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;
import neureka.calculus.environment.implementations.*;
import neureka.calculus.factory.assembly.FunctionBuilder;

public class Summation extends OperationType
{

    public Summation()
    {
        super (
                "summation",
                "sum",
                1,
                false,
                true,
                false,
                true,
                true
        );

        setStringifier(
                children ->
                {
                    String expression = String.join( ", ", children );
                    if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                        return "sum" + expression;
                    }
                    return "sum" + "(" + expression + ")";
                }
        );

        OperationTypeImplementation.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            OperationType type = call.getType();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if (d < 0) {
                    Tsr[] reduction = new Tsr[]{tsrs[0], tsrs[1], tsrs[2]};
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];

                    reduction = AbstractOperationTypeImplementation.Utility._offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];
                } else {
                    tsrs[0] = Tsr.Create.newTsrLike(tsrs[1]).setValue(1.0f);
                }
                return alternative;
            } else {
                return alternative;
            }
        };

        //________________
        // BROADCASTING :

        DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
                (inputs, d) ->
                {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + t2_val[inputs[2].i_of_idx(t2Idx)];
                    else return (t0Idx, t1Idx, t2Idx) -> 1.0;
                };

        Broadcast typeImplementation = new Broadcast(
        ).setADAnalyzer(
                call -> true
        ).setADAgentCreator(
            ( Function f, Tsr derivv, ExecutionCall<Device> call, boolean forward ) ->
            {
                Function mul = Function.Detached.MUL;
                if (
                    derivv != null
                ) {
return new ADAgent(
                            derivv
                   ).withForward(
                            ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, derivv})
                   ).withBackward(
                           null
                   );
                }
                Tsr[] inputs = call.getTensors();
                int d = call.getDerivativeIndex();
                if ( forward )
                {
                    throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                }
                else
                {
                    Tsr deriv = f.derive(inputs, d);
                    return new ADAgent(
                            deriv
).withForward(
                            (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
).withBackward(
                            (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv})
);
                }
            }
        ).setCallHock(
                (caller, call) -> null
        ).setRJAgent(
            rja
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


        setImplementation (
                Broadcast.class,
                typeImplementation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call  ->
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
                                typeImplementation.getKernelSource(), // kernelSource
                                "value = src1 + src2;\n",
                                "value += 1 * drain;\n",
                                this // OperationType
                        )
                )
        );


        //______________
        // ACTIVATION :

        DefaultOperatorCreator<TertiaryNDXConsumer> activationCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if ( d < 0 ) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                    else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                };

        Activation activation = new Activation()
        .setADAnalyzer(
                call -> true
        ).setADAgentCreator(
    ( Function f, Tsr derivv, ExecutionCall<Device> call, boolean forward ) ->
    {
        Function mul = Function.Detached.MUL;
        if (
            derivv != null
        ) {
            return new ADAgent(
                    derivv
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
                    deriv
                ).withForward(
                    ( t, derivative ) -> mul.call(new Tsr[]{derivative, deriv})
                ).withBackward(
                    null
                );
        }
        else
        {
            if ( this.supports(Convolution.class) )
            {
                Function invX = FunctionBuilder.build(
                        "I[0]" + identifier() + ">>I[1]" + identifier() + ">>I[2]",
                        false
                );
                Tsr deriv = f.derive(inputs, d);
                return new ADAgent(
                        deriv
                ).withForward(
                        (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
                ).withBackward(
                        (t, error) -> invX.call(new Tsr[]{error, deriv, new Tsr(t.getPayload().shape(), 0)})
                );
            }
            else
            {
                Tsr deriv = f.derive(inputs, d);
                return new ADAgent(
                            deriv
).withForward(
                            (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
).withBackward(
                            (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv})
);
            }
        }
    }
        ).setCallHock(
                (caller, call) -> null
        ).setRJAgent(
                rja
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

        setImplementation(Activation.class,
                activation.setExecutor(
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
                                                                        activationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = ( call.getTensor(0) != null ) ? 0 : 1;
                                    int gwz =
                                            ( call.getTensor(0) != null )
                                                    ? call.getTensor(0).size()
                                                    : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                activation.getKernelSource(), // kernelSource
                                "output = input;",
                                "output = 1;",
                                this // OperationType
                        )
                )
        );

    }

}
