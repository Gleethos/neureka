package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;

public class CopyLeft extends AbstractOperation {

    public CopyLeft() {
        super(
                new OperationBuilder()
                        .setFunction(         "left_inline"    )
                        .setOperator(         "<"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( false       )
                        .setIsInline(         true      )
        );

        Scalarization scalarization = new Scalarization()
                .setIsSuitableFor(
                        call ->
                        {
                            if ( call.getTsrOfType( Number.class, 1 ).isVirtual() || call.getTsrOfType( Number.class, 1 ).size() == 1 ) {
                                return 1.0f;
                            } else return 0.0f;
                        }
                )
                .setCanPerformBackwardADFor( call -> false )
                .setCanPerformForwardADFor( call -> false )
                .setSupplyADAgentFor(
                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                )
                .setHandleInsteadOfDevice( (caller, call) -> CalcUtil.executeFor( caller, call ) )
                .setInstantiateNewTensorsForExecutionIn(
                        call ->
                        {
                            Tsr[] tsrs = call.getTensors();
                            int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                            call.getTsrOfType( Number.class, offset).incrementVersionBecauseOf(call);
                            call.getTsrOfType( Number.class, offset).setIsVirtual( false );
                            return
                                    ExecutionCall.of(tsrs[offset], tsrs[1+offset])
                                                    .andArgs(Arg.DerivIdx.of(-1))
                                                    .running(this)
                                                    .on( call.getDevice() );
                        }
                )
                .build();

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return t1Idx -> t1_val[ t1Idx.i() ] = value;
                    else return null;
                };

        ScalarOperatorCreator<PrimaryNDAConsumer> scalarXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return t1Idx -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )] = value;
                    else return null;
                };

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call ->
                                {
                                    double value = call.getTsrOfType( Number.class, 1 ).value64( 0 );
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTsrOfType( Number.class, 0 ).size(),
                                                    (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                    ? ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    start, end,
                                                                    scalarXCreator.create(call.getTensors(), value, -1)
                                                            )
                                                    : ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    start, end,
                                                                    scalarCreator.create(call.getTensors(), value, -1)
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
                                                    .passAllOf( t )
                                                    .passAllOf( t )
                                                    .pass( call.getTsrOfType( Number.class, 1 ).value32( 0 ) )
                                                    .pass( t.rank() )
                                                    .pass( call.getDerivativeIndex() )
                                                    .call( gwz );
                                        }
                                )
                                .build()
                )
        );

        Activation activation = new Activation()
            .setCanPerformBackwardADFor( call -> false )
            .setCanPerformForwardADFor( call -> false )
            .setSupplyADAgentFor(
                ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
            )
            .setHandleInsteadOfDevice( (caller, call) -> CalcUtil.executeFor( caller, call ) )
            .setInstantiateNewTensorsForExecutionIn(
                    call ->
                    {
                        Tsr[] tsrs = call.getTensors();
                        int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                        call.getTsrOfType( Number.class, offset).incrementVersionBecauseOf(call);
                        return ExecutionCall.of(tsrs[offset], tsrs[1+offset])
                                            .andArgs(Arg.DerivIdx.of(-1))
                                            .running(Neureka.get().context().getOperation("idy"))
                                            .on(call.getDevice());
                    }
            )
            .build();

        setAlgorithm(
                Activation.class,
                activation
                    .setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call ->
                                {
                                    call.getTsrOfType( Number.class, 0 ).setIsVirtual( false );
                                    Neureka.get().context().getOperation("idy")
                                            .getAlgorithm( Activation.class )
                                            .getImplementationFor( HostCPU.class )
                                            .run(call);
                                },
                                2
                        )
                    )
                    .setImplementationFor(
                        OpenCLDevice.class,
                        call -> {
                            call.getTsrOfType( Number.class, 0 ).setIsVirtual( false );
                            Neureka.get().context().getOperation("idy")
                                    .getAlgorithm(Activation.class)
                                    .getImplementationFor( OpenCLDevice.class )
                                    .run(call);
                        }
                )
        );
    }


    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) reconstructed.append(" <- ");
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
    }
}
