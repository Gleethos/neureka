package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.operations.function.internal.FastFun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class TanhFast extends AbstractActivationOperation
{
    public TanhFast()
    {
        super (
                new OperationBuilder()
                        .setIdentifier(       "fast_tanh"    )
                        .setOperator(         "fast_tanh"    )
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
                                        .with( "output = input * fast_inverse_sqrt( 1.0f + input * input );\n" )
                                        .and(
                                           "float temp1 = input * input;\n" +
                                            "float temp2 = sqrt( 1 + temp1 );\n" +
                                            "output = 1 / ( temp1 * temp2 + temp2 );\n"
                                        )
                        )
        );

    }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected double _activate(double x) { return x * FastFun.invSqrt( 1d + x * x ); }

    @Override protected float _activate(float x) { return x * FastFun.invSqrt( 1f + x * x ); }

    @Override
    protected double _derive( double x ) {
        double temp1 = x * x;
        double temp2 = Math.sqrt( 1 + temp1 );
        return 1 / ( temp1 * temp2 + temp2 );
    }

    @Override
    protected float _derive( float x ) {
        float temp1 = x * x;
        float temp2 = (float) Math.sqrt( 1 + temp1 );
        return 1 / ( temp1 * temp2 + temp2 );
    }

}
