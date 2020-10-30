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

public class ReLU extends AbstractOperationType
{

    private DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        if(t1_val[t1Idx.i()]>=0) return t1_val[t1Idx.i()];
                        else return t1_val[t1Idx.i()]*0.01;
                    };
                } else {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        if(t1_val[t1Idx.i()]>=0) return 1;
                        else return 0.01;
                    };
                }
            };

    private DefaultOperatorCreator<TertiaryNDXConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        if(t1_val[inputs[ 1 ].i_of_idx(t1Idx)]>=0) return t1_val[inputs[ 1 ].i_of_idx(t1Idx)];
                        else return t1_val[inputs[ 1 ].i_of_idx(t1Idx)]*0.01;
                    };
                } else {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        if(t1_val[inputs[ 1 ].i_of_idx(t1Idx)]>=0) return 1;
                        else return 0.01;
                    };
                }
            };

    public ReLU()
    {
        super(
                "relu",
                "relu",
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
                        return "relu" + expression;
                    }
                    return "relu" + "(" + expression + ")";
                }
        );

        Activation typeImplementation = new Activation()
        .setBackwardADAnalyzer( call -> true )
        .setForwardADAnalyzer(
                call -> {
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        ).setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
            defaultImplementation().supplyADAgentFor(f, call, forward)
        )
        .setCallHock( ( caller, call ) -> null )
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[ 1 ].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual( false );
                        try {
                            device.store(output);
                        } catch( Exception e ) {
                            e.printStackTrace();
                        }
                        tsrs[ 0 ] = output;
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
                                "if (input >= 0) {  output = input; } else { output = input * (float)0.01; }\n",
                                "if (input >= 0) { output = (float)1; } else { output = (float)0.01; }\n",
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
        double output;
        if ( !derive ) {
            if ( input >= 0 ) output = input;
            else output = input * 0.01;
        } else {
            if ( input >= 0 ) output = 1;
            else output = 0.01;
        }
        return output;
    }


}
