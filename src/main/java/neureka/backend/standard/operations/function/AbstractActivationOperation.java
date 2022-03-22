package neureka.backend.standard.operations.function;

import neureka.Tsr;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

abstract class AbstractActivationOperation extends AbstractOperation {

    AbstractActivationOperation(String identifier)
    {
        super(
            new OperationBuilder()
                .setIdentifier(       identifier    )
                .setOperator(         identifier    )
                .setArity(            1             )
                .setIsOperator(       false         )
                .setIsIndexer(        false         )
                .setIsDifferentiable( true          )
                .setIsInline(         false         )
        );
        setAlgorithm(
            new Activation().setSupplyADAgentFor( getDefaultAlgorithm() ).buildFunAlgorithm()
                .setImplementationFor(
                    CPU.class,
                    Activation.implementationForCPU()
                        .with(Fun.F64ToF64.pair(this::_activate, this::_derive))
                        .with(Fun.F32ToF32.pair(this::_activate, this::_derive))
                        .with(Fun.I32ToI32.pair(this::_activate, this::_derive))
                        .get()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                            .with( _activationCode() )
                            .and( _derivationCode() )
                )
        );

        setAlgorithm(
            new Scalarization()
            .setCanPerformBackwardADFor( call -> true )
            .setCanPerformForwardADFor(
                call -> {
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.inputs() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
            )
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
            .setCallPreparation(
                call -> {
                    Device device = call.getDevice();
                    if ( call.input( 0 ) == null ) // Creating a new tensor:
                    {
                        int[] shp = call.input( 1 ).getNDConf().shape();
                        Tsr<?> output = Tsr.of( shp, 0.0 ).getUnsafe().setIsIntermediate( true );
                        output.setIsVirtual( false );
                        try {
                            device.store( output );
                        } catch( Exception e ) {
                            e.printStackTrace();
                        }
                        call.setInput( 0, output );
                    }
                    return call;
                }
            )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                Scalarization.implementationForCPU()
                        .with(Fun.F64F64ToF64.triple(
                                ( a, b ) -> _activate(b),
                                ( a, b ) -> _derive(b), // Deriving at input 0
                                ( a, b ) -> _derive(b)  // Deriving input 1
                        ))
                        .with(Fun.F32F32ToF32.triple(
                                ( a, b ) -> _activate(b),
                                ( a, b ) -> _derive(b), // Deriving at input 0
                                ( a, b ) -> _derive(b)  // Deriving input 1
                        ))
                        .with(Fun.I32I32ToI32.triple(
                                ( a, b ) -> _activate(b),
                                ( a, b ) -> _derive(b), // Deriving at input 0
                                ( a, b ) -> _derive(b)  // Deriving input 1
                        ))
                        .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation.compiler()
                    .arity( 2 )
                    .kernelSource( Scalarization.getKernelSource() )
                    .activationSource( _activationCode() )
                    .differentiationSource( _derivationCode() )
                    .kernelPostfix( this.getIdentifier() )
                    .execution(
                        call -> {
                            Tsr<Number> t = call.getTsrOfType( Number.class, 0 );
                            int gwz = t.size();
                            call.getDevice()
                                    .getKernel(call)
                                    .passAllOf(t)
                                    .passAllOf(t)
                                    .pass((float)call.getTsrOfType( Number.class, 1 ).getDataAs( double[].class )[ 0 ])
                                    .pass(t.rank())
                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                    .call( gwz );
                        }
                    )
                    .build()
            )
        );
    }

    @Override
    public final String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return getIdentifier() + expression;
        return getIdentifier() + "(" + expression + ")";
    }

    @Override
    public final double calculate( double[] inputs, int j, int d, Function[] src ) {
        boolean derive = d >= 0;
        double inner = ( !derive ? 1 : src[ 0 ].derive( inputs, d, j ) );
        return calculate( src[ 0 ].call( inputs, j ),  derive ) * inner;
    }

    @Contract(pure = true)
    private double calculate(double input, boolean derive ) {
        if ( !derive )
            return _activate( input );
        else
            return _derive( input ) ;
    }

    protected abstract String _activationCode();

    protected abstract String _derivationCode();

    protected abstract double _activate(double x);

    protected float _activate(float x) { return (float) _activate( (double) x ); }

    protected int _activate(int x) { return (int) Math.round( _activate( (double) x ) ); }

    protected abstract double _derive(double x);

    protected float _derive(float x) { return (float) _derive( (double) x ); }

    protected int _derive(int x) { return (int) Math.round( _derive( (double) x ) ); }

}
