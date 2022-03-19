package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class TanhQuick extends AbstractActivationOperation
{
    public TanhQuick()
    {
        super (
                new OperationBuilder()
                        .setIdentifier(       "quick_tanh"    )
                        .setOperator(         "quick_tanh"    )
                        .setArity(            1         )
                        .setIsOperator(       false     )
                        .setIsIndexer(        false     )
                        .setIsDifferentiable( true      )
                        .setIsInline(         false     )
        );
        setAlgorithm(
                new Activation()
                        .setSupplyADAgentFor( getDefaultAlgorithm() )
                        .buildFunAlgorithm()
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
                                        .with( "output = input / ( 1.0f + fabs( input ) );\n" )
                                        .and("output = 1.0f / ( 2.0f * fabs( input ) + input * input + 1.0f );\n")
                        )
        );

    }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected double _activate(double x) { return x / ( 1d + Math.abs( x ) ); }

    @Override protected float _activate(float x) { return x / ( 1f + Math.abs( x ) ); }

    @Override protected double _derive(double x) { return 1d / ( 2d * Math.abs( x ) + x * x + 1d ); }

    @Override protected float _derive(float x) { return 1f / ( 2f * Math.abs( x ) + x * x + 1f ); }

}
