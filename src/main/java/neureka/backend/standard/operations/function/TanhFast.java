package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.operations.function.internal.FastFun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public class TanhFast extends AbstractOperation
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
                                        .with(Fun.F64ToF64.pair(
                                                x -> x * FastFun.invSqrt( 1d + x * x ),
                                                x -> _derive( x )
                                        ))
                                        .with(Fun.F32ToF32.pair(
                                                x -> ( x * FastFun.invSqrt( 1f + x * x ) ),
                                                x -> _derive( x )
                                        ))
                                        .with(Fun.I32ToI32.pair(
                                                x -> Math.round( x * FastFun.invSqrt( 1f + x * x ) ),
                                                x -> Math.round( _derive( x ) )
                                        ))
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
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return getIdentifier() + expression;
        return getIdentifier() + "(" + expression + ")";
    }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
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
        if ( !derive ) return input * FastFun.invSqrt(1d + input * input );
        else return _derive( input ) ;
    }

    private static double _derive( double x ) {
        double temp1 = x * x;
        double temp2 = Math.sqrt( 1 + temp1 );
        return 1 / ( temp1 * temp2 + temp2 );
    }

    private static float _derive( float x ) {
        float temp1 = x * x;
        float temp2 = (float) Math.sqrt( 1 + temp1 );
        return 1 / ( temp1 * temp2 + temp2 );
    }

}
