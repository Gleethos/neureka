package neureka.calculus.environment.operations.function;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.AbstractOperationType;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.implementations.*;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Absolute extends AbstractOperationType {

    private DefaultOperatorCreator<TertiaryNDXConsumer> _activationCreator =
    (inputs, d)->{
        double[] t1_val = inputs[1].value64();
        if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.abs(t1_val[inputs[1].i_of_idx(t1Idx)]);
        else return (t0Idx, t1Idx, t2Idx) -> (t1_val[inputs[1].i_of_idx(t1Idx)] < 0) ? -1 : 1;
    };

    public Absolute()
    {
        super("abs", "abs" , 1, false, false, true, true);

        setStringifier(
            children -> {
                String expression = String.join( ", ", children );
                if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                    return "abs" + expression;
                }
                return "abs" + "(" + expression + ")";
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
    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
            {
                Tsr derivv = (Tsr)call.getAt("derivative");
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

    @Override
    public double calculate(double[] inputs, int j, int d, List<Function> src) {
        return calculate(
                src.get(0).call( inputs, j ),
                d >= 0
        ) * ( ( d < 0 ) ? 1 : src.get(0).derive( inputs, d, j ) );
    }

    @Contract(pure = true)
    public static double calculate(double input, boolean derive ) {
        if ( !derive ) return Math.abs( input );
        else return ( input < 0 ) ? -1 : 1;
    }



}
