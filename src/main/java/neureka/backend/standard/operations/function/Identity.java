package neureka.backend.standard.operations.function;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Identity extends AbstractOperation
{

    public Identity()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "idy"    )
                        .setOperator(         "idy"    )
                        .setArity(            1        )
                        .setIsOperator(       false    )
                        .setIsIndexer(        false    )
                        .setIsDifferentiable( true     )
                        .setIsInline(         false    )
        );

        DefaultOperatorCreator<TertiaryNDIConsumer> activationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ];
                    else return ( t0Idx, t1Idx, t2Idx ) -> 1;
                };

        DefaultOperatorCreator<TertiaryNDAConsumer> activationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                    else return ( t0Idx, t1Idx, t2Idx ) -> 1;
                };

        Activation operationAlgorithm = new Activation()
        .setCanPerformBackwardADFor( call -> true )
        .setCanPerformForwardADFor(
                call -> {
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        ).setSupplyADAgentFor(
            ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
        )
        .setHandleInsteadOfDevice( (caller, call ) -> null )
        .setHandleRecursivelyAccordingToArity( (call, goDeeperWith ) -> null )
        .setInstantiateNewTensorsForExecutionIn(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return ExecutionCall.builder()
                                .device(call.getDevice())
                                .tensors(new Tsr[]{tsrs[offset], tsrs[1+offset]})
                                .derivativeIndex(-1)
                                .operation(Neureka.get().context().instance("idy"))
                                .build();
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
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        start, end,
                                                                        activationXCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ),
                                                                        start, end,
                                                                        activationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                2
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 2 )
                                .kernelSource( operationAlgorithm.getKernelSource() )
                                .activationSource( "output = input;\n" )
                                .differentiationSource( "output = input;\n" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                            // Drain tensor needs to be 'actual'! :
                                            if (!call.getTsrOfType( Number.class, offset + 1).isVirtual()) call.getTsrOfType( Number.class, offset).setIsVirtual( false );
                                            call.getDevice().getKernel(call)
                                                    .pass( call.getTsrOfType( Number.class, offset ) )
                                                    .pass( call.getTsrOfType( Number.class, offset + 1 ) )
                                                    .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                    .pass( call.getDerivativeIndex() )
                                                    .call( gwz );
                                        }
                                )
                                .build()
                )
        );

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarizationCreator =
                (inputs, value, d) -> {
                    if ( d < 0 ) return t1Idx -> value;
                    else return t1Idx -> value;
                };
        Scalarization scalarization = new Scalarization()
            .setCanPerformBackwardADFor( call -> true )
            .setCanPerformForwardADFor(
                    call -> {
                        Tsr<?> last = null;
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                    }
            )
            .setSupplyADAgentFor(
                ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
            )
            .setHandleInsteadOfDevice( (caller, call ) -> null )
            .setHandleRecursivelyAccordingToArity( (call, goDeeperWith ) -> null )
            .setInstantiateNewTensorsForExecutionIn(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[ 1 ].getNDConf().shape();
                        Tsr output = Tsr.of( shp, 0.0 );
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
                                    double value = call.getTsrOfType( Number.class, 0 ).value64( 2 );
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (start, end) ->
                                                                Scalarization.scalarize(
                                                                        call.getTsrOfType( Number.class, 0 ), start, end,
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
                        CLImplementation.compiler()
                                .arity( 2 )
                                .kernelSource( scalarization.getKernelSource() )
                                .activationSource( "output = value;\n" )
                                .differentiationSource( "output = value;\n" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            Tsr t = call.getTsrOfType( Number.class, 0 );
                                            int gwz = t.size();
                                            call.getDevice().getKernel(call)
                                                    .pass(t)
                                                    .pass(t)
                                                    .pass((float)call.getTsrOfType( Number.class, 1 ).value64( 0 ))
                                                    .pass(t.rank())
                                                    .pass( call.getDerivativeIndex() )
                                                    .call( gwz );
                                        }
                                )
                                .build()
                )
        );


    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "idy" + expression;
        return "idy" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return calculate(
                src[ 0 ].call( inputs, j ),
                d >= 0
        ) * ( ( d < 0 ) ? 1 : src[ 0 ].derive( inputs, d, j ) );
    }

    @Contract(pure = true)
    public static double calculate(double input, boolean derive) {
        if ( !derive ) return input;
        else return 1;
    }



}
