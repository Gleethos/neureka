package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Sigmoid extends AbstractActivationOperation
{
    public Sigmoid()
    {
        super(
            new OperationBuilder()
                .setIdentifier(       "sig"    )
                .setOperator(         "sig"    )
                .setArity(            1        )
                .setIsOperator(       false    )
                .setIsIndexer(        false    )
                .setIsDifferentiable( true     )
                .setIsInline(         false    )
        );
        setAlgorithm(
            Activation.class,
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
                            .with( "output = 1 / ( 1 + (float) exp(-input) );\n" )
                            .and( "output = input * ( 1 - input );\n" )
            )
        );
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected double _activate(double x) { return sig(x); }

    @Override protected float _activate(float x) { return (float) sig(x); }

    @Override
    protected double _derive(double x) {
        double sig = _activate(x);
        return sig * ( 1 - sig );
    }

    @Override
    protected float _derive(float x) {
        float sig = _activate(x);
        return sig * ( 1 - sig );
    }

    public static double sig(double x) { return 1d / ( 1d + Math.exp( -x ) ); }

}




