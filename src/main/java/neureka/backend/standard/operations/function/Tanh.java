package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Tanh extends AbstractOperation
{
    public Tanh()
    {
        super (
            new OperationBuilder()
                .setFunction(         "tanh"    )
                .setOperator(         "tanh"    )
                .setArity(            1         )
                .setIsOperator(       false     )
                .setIsIndexer(        false     )
                .setIsDifferentiable( true      )
                .setIsInline(         false     )
        );

        Activation operationAlgorithm = new Activation()
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .buildFunAlgorithm();

        setAlgorithm(
            Activation.class,
            operationAlgorithm.setImplementationFor(
                CPU.class,
                Activation.implementationForCPU()
                    .with(Fun.F64ToF64.pair(
                        x -> Math.tanh( x ),
                        x -> 1 - Math.pow( Math.tanh( x ), 2 )
                    ))
                    .with(Fun.F32ToF32.pair(
                        x -> (float) Math.tanh( x ),
                        x -> (float) ( 1 - Math.pow( Math.tanh( x ), 2 ) )
                    ))
                    .with(Fun.I32ToI32.pair(
                        x -> (int) Math.round( Math.tanh( x ) ),
                        x -> (int) Math.round( 1 - Math.pow( Math.tanh( x ), 2 ) )
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                            .with( "output = input/pow(1+pow(input, 2.0f), 0.5f);\n" )
                            .and( "output = 1-pow(input/pow((1.0f+pow(input,2.0f)),0.5f), 2.0f);\n" )
            )
        );

    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "tanh" + expression;
        return "tanh" + "(" + expression + ")";
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
        final double pow = Math.pow((1 + Math.pow(input, 2)), 0.5);
        if ( !derive ) {
            return input / pow;
        } else {
            return (1 - Math.pow((input / pow), 2));
        }
    }

}

