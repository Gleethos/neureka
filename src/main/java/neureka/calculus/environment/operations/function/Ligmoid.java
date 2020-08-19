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


public class Ligmoid extends OperationType
{

    private DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
            (inputs, d)->{
                double[] t1_val = inputs[1].value64();
                if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.log(1 + Math.pow(Math.E, t1_val[inputs[1].i_of_idx(t1Idx)]));
                else return (t0Idx, t1Idx, t2Idx) -> 1 / (1 + Math.pow(Math.E, -t1_val[inputs[1].i_of_idx(t1Idx)]));
            };

    public Ligmoid()
    {
        super("ligmoid", "lig" , 1, false, false, false, true, true);

        setStringifier(
                children -> {
                    String expression = String.join( ", ", children );
                    if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                        return "lig" + expression;
                    }
                    return "lig" + "(" + expression + ")";
                }
        );

        Activation typeImplementation = new Activation()
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
                                "output = \n" +
                                        "   (\n" +
                                        "        (float) log(\n" +
                                        "            1+pow(\n" +
                                        "                (float)\n" +
                                        "                M_E,\n" +
                                        "                (float)\n" +
                                        "                input\n" +
                                        "            )\n" +
                                        "        )\n" +
                                        "    );",
                                "output =\n" +
                                        "    1 /\n" +
                                        "        (1 + (float) pow(\n" +
                                        "                (float)M_E,\n" +
                                        "                (float)input\n" +
                                        "            )\n" +
                                        "        );\n",
                                this // OperationType
                        )
                )
        );



    }


}
