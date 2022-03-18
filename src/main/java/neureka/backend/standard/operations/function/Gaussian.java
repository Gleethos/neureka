package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Gaussian extends AbstractOperation
{
    public Gaussian()
    {
        super(
            new OperationBuilder()
                    .setIdentifier(         "gaus"    )
                    .setOperator(         "gaus"    )
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
                    .with(Fun.F64ToF64.pair(
                        x -> Math.exp( -( x * x ) ),
                        x -> -2 * x * Math.exp( -( x * x ) )
                    ))
                    .with(Fun.F32ToF32.pair(
                        x -> (float) Math.exp( -( x * x ) ),
                        x -> (float) ( -2 * x * Math.exp( -( x * x ) ) )
                    ))
                    .with(Fun.I32ToI32.pair(
                        x -> (int) Math.round( Math.exp( -( x * x ) ) ),
                        x -> (int) Math.round( -2 * x * Math.exp( -( x * x ) ) )
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                        .with("output = exp( -( input * input ) );\n" )
                        .and("output = -2 * input * exp( -( input * input ) );\n")
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
        if ( !derive ) return Math.pow(Math.E, -Math.pow(input, 2));
        else return -2 * input * Math.pow(Math.E, -Math.pow(input, 2));
    }



}
