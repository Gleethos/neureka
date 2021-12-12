package neureka.backend.standard.operations.function;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Sinus extends AbstractOperation
{

    private final DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].getDataAs( double[].class );
                if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> Math.sin(t1_val[ t1Idx.i() ]);
                else return ( t0Idx, t1Idx, t2Idx ) -> Math.cos(t1_val[ t1Idx.i() ]);
            };

    private final DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].getDataAs( double[].class );
                if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> Math.sin(t1_val[inputs[ 1 ].indexOfIndices( t1Idx )]);
                else return ( t0Idx, t1Idx, t2Idx ) -> Math.cos(t1_val[inputs[ 1 ].indexOfIndices( t1Idx )]);
            };

    public Sinus()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "sin"    )
                        .setOperator(         "sin"    )
                        .setArity(            1        )
                        .setIsOperator(       false    )
                        .setIsIndexer(        false    )
                        .setIsDifferentiable( true     )
                        .setIsInline(         false    )
        );

        Activation operationAlgorithm =
                new Activation()
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
                        .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
                        .setCallPreparation(
                             call -> {
                                 Tsr<?>[] tsrs = call.getTensors();
                                 Device device = call.getDevice();
                                 if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                                 {
                                     int[] shp = tsrs[ 1 ].getNDConf().shape();
                                     Tsr<?> output = Tsr.of( shp, 0.0 );
                                     output.setIsVirtual( false );
                                     try {
                                         device.store( output );
                                     } catch ( Exception e ) {
                                         e.printStackTrace();
                                     }
                                     tsrs[ 0 ] = output;
                                 }
                                 return call;
                             }
                        )
                        .buildFunAlgorithm();

        setAlgorithm(
                Activation.class,
                operationAlgorithm.setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        start, end,
                                                                        _creatorX.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ),
                                                                        start, end,
                                                                        _creator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                )
                            )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operationAlgorithm.getKernelSource() )
                                .activationSource( "output = sin( input );\n" )
                                .differentiationSource( "output = cos( input );\n" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                            call.getDevice().getKernel(call)
                                                    .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                                    .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                                    .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
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
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "sin" + expression;
        return "sin" + "(" + expression + ")";
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
    public static double calculate(double input, boolean derive ) {
        if ( !derive ) return Math.sin( input );
        else return Math.cos( input );
    }




}
