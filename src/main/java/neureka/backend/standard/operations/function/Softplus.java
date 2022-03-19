package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;


public final class Softplus extends AbstractOperation
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
                    .with(Fun.F64ToF64.pair(
                        x -> Math.log( 1d + Math.exp( x ) ),
                        x -> 1d / ( 1d + Math.exp( -x ) )
                    ))
                    .with(Fun.F32ToF32.pair(
                        x -> (float) Math.log(1 + Math.exp( x ) ),
                        x -> (float) ( 1f / ( 1f + Math.exp( -x ) ) )
                    ))
                    .with(Fun.I32ToI32.pair(
                        x -> (int) Math.round( Math.log( 1 + Math.exp( x ) ) ),
                        x -> (int) Math.round( 1d / ( 1d + Math.exp( -x ) ) )
                    ))
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
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return getIdentifier() + expression;
        return getIdentifier() + "(" + expression + ")";
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
        if ( !derive ) return Math.log(1 + Math.exp( input));
        else return Sigmoid.sig(input);
    }



}
