package neureka.backend.standard.operations.function;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.api.operations.Operation;
import neureka.devices.Device;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.calculus.Function;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.api.ExecutionCall;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Identity extends AbstractOperation
{

    public Identity()
    {
        super("idy", "idy" , 1, false, false, true, false);

        setStringifier(
                children -> {
                    String expression = String.join( ", ", children );
                    if ( expression.startsWith("(") && expression.endsWith(")") ) return "idy" + expression;
                    return "idy" + "(" + expression + ")";
                }
        );

        DefaultOperatorCreator<TertiaryNDIConsumer> activationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                    else return ( t0Idx, t1Idx, t2Idx ) -> 1;
                };

        DefaultOperatorCreator<TertiaryNDXConsumer> activationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].i_of_idx( t1Idx )];
                    else return ( t0Idx, t1Idx, t2Idx ) -> 1;
                };

        Activation operationAlgorithm = new Activation()
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
                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
        )
        .setCallHook( (caller, call ) -> null )
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, Operation.instance("idy") );
                }
        )
        .build();

        setAlgorithm(
                Activation.class,
                operationAlgorithm.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor( 0 ),
                                                                        start, end,
                                                                        activationXCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTensor( 0 ), call.getTensor( 1 ),
                                                                        start, end,
                                                                        activationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                2
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = (call.getTensor( 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTensor( 0 ) != null) ? call.getTensor( 0 ).size() : call.getTensor( 1 ).size();
                                    // Drain tensor needs to be 'actual'! :
                                    if (!call.getTensor(offset + 1).isVirtual()) call.getTensor(offset).setIsVirtual( false );
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTensor( offset ) )
                                            .pass( call.getTensor( offset + 1 ) )
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                2,
                                operationAlgorithm.getKernelSource(), // kernelSource
                                "output = input;\n", // activationSource
                                "output = input;\n", //differentiationSource
                                this // OperationType
                        )
                )
        );

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarizationCreator =
                (inputs, value, d) -> {
                    if ( d < 0 ) return t1Idx -> value;
                    else return t1Idx -> value;
                };
        Scalarization scalarization = new Scalarization()
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
            )
            .setADAgentSupplier(
                ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
            )
            .setCallHook( (caller, call ) -> null )
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
            )
            .build();

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call  -> {
                                    double value = call.getTensor( 0 ).value64(2);
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        (start, end) ->
                                                                Scalarization.scalarize(
                                                                        call.getTensor( 0 ), start, end,
                                                                        scalarizationCreator.create(
                                                                                call.getTensors(), value, call.getDerivativeIndex()
                                                                        )
                                                                )
                                                );
                                },
                                2
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    Tsr t = call.getTensor( 0 );
                                    int gwz = t.size();
                                    call.getDevice().getKernel(call)
                                            .pass(t)
                                            .pass(t)
                                            .pass((float)call.getTensor( 1 ).value64( 0 ))
                                            .pass(t.rank())
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                2,
                                scalarization.getKernelSource(), // kernelSource
                                "output = value;\n",
                                "output = value;\n",
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
    public static double calculate(double input, boolean derive) {
        if ( !derive ) return input;
        else return 1;
    }



}
