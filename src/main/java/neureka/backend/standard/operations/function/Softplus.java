package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;


public final class Softplus extends AbstractActivationOperation
{
    public Softplus()
    {
        super(
            new OperationBuilder()
                    .setIdentifier(         "softplus"    )
                    .setOperator(         "softplus"    )
                    .setArity(            1             )
                    .setIsOperator(       false         )
                    .setIsIndexer(        false         )
                    .setIsDifferentiable( true          )
                    .setIsInline(         false         )
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
                            .with("output = log( 1.0f + exp( input ) );")
                            .and("output = 1.0f / ( 1.0f + exp( -input ) );\n")
            )
        );
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected double _activate(double x) { return Math.log( 1d + Math.exp( x ) ); }

    @Override protected double _derive(double x) { return 1d / ( 1d + Math.exp( -x ) ); }

}
