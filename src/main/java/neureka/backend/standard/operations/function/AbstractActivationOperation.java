package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.ScalarActivation;
import neureka.backend.standard.algorithms.ScalarBroadcast;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.NDimensional;
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
            new ScalarBroadcast(Fun.F64ToF64.pair(this::_activate, this::_derive))
            .setCanPerformBackwardADFor( call -> true )
            .setCanPerformForwardADFor(
                call -> call.validate().allNotNullHaveSame(NDimensional::shape).isValid()
            )
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                ScalarBroadcast.implementationForCPU()
                        .with(Fun.F64ToF64.pair(this::_activate, this::_derive))
                        .with(Fun.F32ToF32.pair(this::_activate, this::_derive))
                        .with(Fun.I32ToI32.pair(this::_activate, this::_derive))
                        .get()
            )
        );

        setAlgorithm(
            new ScalarActivation(Fun.F64ToF64.pair(this::_activate, this::_derive))
            .setCanPerformBackwardADFor( call -> true )
            .setCanPerformForwardADFor(
                call -> call.validate().allNotNullHaveSame(NDimensional::shape).isValid()
            )
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
            .buildFunAlgorithm()
            .setImplementationFor(
                    CPU.class,
                    ScalarActivation.implementationForCPU()
                            .with(Fun.F64ToF64.pair(this::_activate, this::_derive))
                            .with(Fun.F32ToF32.pair(this::_activate, this::_derive))
                            .with(Fun.I32ToI32.pair(this::_activate, this::_derive))
                            .get()
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
