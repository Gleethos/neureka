package neureka.calculus.environment.operations.other;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.*;
import neureka.calculus.factory.assembly.FunctionBuilder;

import java.util.Random;

public class Randomization extends OperationType
{

    public Randomization()
    {
        super(
                "random", "rand", 1,
                true, false, false, false, false
        );

        ScalarOperatorCreator< PrimaryNDXConsumer > creator =
                ( inputs, value, d ) -> {
                    //double[] t1_val = inputs[1].value64();
                    return t1Idx -> {
                        int sum = 0;
                        for (int idx : t1Idx) sum += idx;
                        Random dice = new Random();
                        dice.setSeed(Double.doubleToLongBits(value+sum));
                        return dice.nextGaussian();
                    };
                };

        Scalarization scalarization = new Scalarization()
        .setADAnalyzer(
                call -> {
                    if ( call.getType().supports(Convolution.class) ) return false;
                    if ( call.getType().identifier().equals(",") ) return false; //Reshape
                    Tsr last = null;
                    for ( Tsr t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        ).setADAgentCreator(
    ( Function f, Tsr derivv, ExecutionCall<Device> call, boolean forward ) ->
    {
        Function mul = Function.Detached.MUL;
        if (
            derivv != null
        ) {
            return new ADAgent(
                    () -> derivv,
                    ( t, derivative ) -> mul.call(new Tsr[]{derivative, derivv}),
                    null
            );
        }
        Tsr[] inputs = call.getTensors();
        int d = call.getDerivativeIndex();
        if( forward )
        {
            Tsr deriv = f.derive(inputs, d);
            return new ADAgent(
                    () -> deriv,
                    ( t, derivative ) -> mul.call(new Tsr[]{derivative, deriv}),
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
                        ()->deriv,
                        (t, derivative) -> mul.call(new Tsr[]{derivative, deriv}),
                        (t, error) -> invX.call(new Tsr[]{error, deriv, new Tsr(t.getPayload().shape(), 0)})
                );
            }
            else
            {
                Tsr deriv = f.derive(inputs, d);
                return new ADAgent(
                        ()->deriv,
                        (t, derivative) -> mul.call(new Tsr[]{derivative, deriv}),
                        (t, error) -> mul.call(new Tsr[]{error, deriv})
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
                    int offset = ( tsrs[0] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                }
        );

        setImplementation(
                Scalarization.class,
                scalarization.setExecutor(
                        HostExecutor.class,
                        new HostExecutor (
                                call -> call.getDevice().getExecutor()
                                        .threaded (
                                                call.getTensor(0).size(),
                                                ( start, end ) ->
                                                        Scalarization.scalarize (
                                                                call.getTensor(0),
                                                                start, end,
                                                                creator.create(
                                                                        call.getTensors(),
                                                                        call.getTensor(1).value64(0),
                                                                        call.getDerivativeIndex()
                                                                )
                                                        )
                                        ),
                                3
                        )
                )
        );

    }

}
