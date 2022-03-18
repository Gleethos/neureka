package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Logarithm extends AbstractOperation
{
    public Logarithm()
    {
        super (
            new OperationBuilder()
                    .setFunction(         "ln"  )
                    .setOperator(         "ln"  )
                    .setArity(            1      )
                    .setIsOperator(       false  )
                    .setIsIndexer(        false  )
                    .setIsDifferentiable( true   )
                    .setIsInline(         false  )
        );
        setAlgorithm(
            new Activation()
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                Activation.implementationForCPU()
                    .with(Fun.F64ToF64.pair(
                        x -> Math.log(x),
                        x -> 1d / x
                    ))
                    .with(Fun.F32ToF32.pair(
                        x -> (float) Math.log(x),
                        x -> 1f / x
                    ))
                    .with(Fun.I32ToI32.pair(
                            x -> (int) Math.round(Math.log(x)),
                            x -> (int) Math.round( 1d / x )
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                            .with( "output = log( input );\n" )
                            .and( "output = 1.0 / ( input );\n" )
            )
        );

    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "ln" + expression;
        return "ln" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        if ( children.length != 1 ) throw new IllegalStateException("Natual logarithm does not support more than 1 argument.");
        return children[0].getDerivative(derivationIndex)+" / "+children[0].toString();
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return calculate(
                src[ 0 ].call( inputs, j ),
                d >= 0
        ) * ( ( d < 0 ) ? 1 : src[ 0 ].derive( inputs, d, j ) );
    }

    @Contract(pure = true)
    public static double calculate( double input, boolean derive ) {
        if ( !derive ) return Math.log( input );
        else return 1.0 / input;
    }


}
