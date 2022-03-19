package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public final class Tanh extends AbstractActivationOperation
{
    public Tanh()
    {
        super (
            new OperationBuilder()
                .setIdentifier(         "tanh"    )
                .setOperator(         "tanh"    )
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
                        .with( "output = tanh(input);\n" )
                        .and( "output = 1 - pow( tanh(input), 2.0f );\n" )
            )
        );

    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected double _activate( double x ) { return 2 / ( 1 + Math.exp( -x * 2 ) ) - 1; }

    @Override protected float _activate( float x ) { return (float) (2 / ( 1 + Math.exp( -x * 2 ) ) - 1); }

    @Override protected double _derive( double x ) { return  1 - Math.pow( _tanh( x ), 2 ); }

    @Override protected float _derive( float x ) { return (float) (1 - Math.pow( _tanh( x ), 2 )); }

    private static double _tanh( double x ) { return 2 / ( 1 + Math.exp( -x * 2 ) ) - 1; }

    private static float _tanh( float x ) { return (float) (2 / ( 1 + Math.exp( -x * 2 ) ) - 1); }

}

