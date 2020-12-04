package neureka.calculus.backend.operations.function;

import neureka.Neureka;
import neureka.Tsr;
import neureka.devices.Device;
import neureka.devices.host.execution.HostExecutor;
import neureka.devices.opencl.execution.CLExecutor;
import neureka.calculus.Function;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.implementations.functional.Activation;
import neureka.calculus.backend.operations.AbstractOperationType;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Sigmoid extends AbstractOperationType
{

    private DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                if (d < 0) return (t0Idx, t1Idx, t2Idx) -> calculate( t1_val[t1Idx.i()], false );
                else return (t0Idx, t1Idx, t2Idx) ->  calculate( t1_val[t1Idx.i()], true );
            };

    private DefaultOperatorCreator<TertiaryNDXConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                if (d < 0) return (t0Idx, t1Idx, t2Idx) -> calculate( t1_val[inputs[ 1 ].i_of_idx(t1Idx)], false );
                else return (t0Idx, t1Idx, t2Idx) ->  calculate( t1_val[inputs[ 1 ].i_of_idx(t1Idx)], true );
            };


    public Sigmoid()
    {
        super(
                "sig",
                "sig" ,
                1,
                false,
                false,
                true,
                false
        );

        setStringifier(
                children -> {
                    String expression = String.join( ", ", children );
                    if (expression.charAt( 0 ) == '(' && expression.charAt(expression.length() - 1) == ')') {
                        return "sig" + expression;
                    }
                    return "sig" + "(" + expression + ")";
                }
        );

        Activation typeImplementation = new Activation()
            .setADAgentSupplier(
                ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                defaultImplementation().supplyADAgentFor(f, call, forward)
            );


        setImplementation(
                Activation.class,
                typeImplementation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor( 0 ),
                                                                        start, end,
                                                                        _creatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor( 0 ), call.getTensor( 1 ),
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
                                    int offset = (call.getTensor( 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTensor( 0 ) != null) ? call.getTensor( 0 ).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTensor( offset ) )
                                            .pass( call.getTensor( offset + 1 ) )
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                typeImplementation.getKernelSource(), // kernelSource
                                "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                                "output = input * (1 - input);\n",
                                this // OperationType
                        )
                )
        );




    }

    @Override
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
        return calculate(
                src.get( 0 ).call( inputs, j ),
                d >= 0
        ) * ( ( d < 0 ) ? 1 : src.get( 0 ).derive( inputs, d, j ) );
    }

    @Contract(pure = true)
    public static double calculate(double input, boolean derive ) {
        if ( !derive ) return 1 / (1 + Math.pow(Math.E, -input));
        else {
            double sig = calculate(input, false);
            return sig * (1 - sig);
        }
    }


}




