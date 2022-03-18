package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public class TanhQuick extends AbstractOperation
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
                                        .with(Fun.F64ToF64.pair(
                                                x -> x / ( 1d + Math.abs( x ) ),
                                                x -> 1d / ( 2d * Math.abs( x ) + x * x + 1d )
                                        ))
                                        .with(Fun.F32ToF32.pair(
                                                x -> x / ( 1f + Math.abs( x ) ),
                                                x -> 1f / ( 2f * Math.abs( x ) + x * x + 1f )
                                        ))
                                        .with(Fun.I32ToI32.pair(
                                                x -> Math.round( x / ( 1f + Math.abs( x ) ) ),
                                                x -> Math.round( 1f / ( 2f * Math.abs( x ) + x * x + 1f ) )
                                        ))
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
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "tanh" + expression;
        return getIdentifier() + "(" + expression + ")";
    }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return calculate(src[ 0 ].call( inputs, j ), d >= 0) * ( ( d < 0 ) ? 1 : src[ 0 ].derive( inputs, d, j ) );
    }

    @Contract(pure = true)
    public static double calculate(double input, boolean derive ) {
        if ( !derive ) return input / ( 1d + Math.abs( input ) );
        else return 1d / ( 2d * Math.abs( input ) + input * input + 1d ) ;
    }

}
