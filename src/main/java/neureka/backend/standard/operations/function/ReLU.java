package neureka.backend.standard.operations.function;

import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class ReLU extends AbstractOperation
{
    public ReLU()
    {
        super(
            new OperationBuilder()
                    .setIdentifier(         "relu"    )
                    .setOperator(         "relu"    )
                    .setArity(            1        )
                    .setIsOperator(       false    )
                    .setIsIndexer(        false    )
                    .setIsDifferentiable( true     )
                    .setIsInline(         false    )
        );
        setAlgorithm(
            new Activation()
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                Activation.implementationForCPU()
                    .with(Fun.F64ToF64.pair(
                        x -> (  x >= 0 ? x : x * .01 ),
                        x -> (  x >= 0 ? 1 :  .01    )
                    ))
                    .with(Fun.F32ToF32.pair(
                        x -> (  x >= 0 ? x  : x * .01f ),
                        x -> (  x >= 0 ? 1f : .01f     )
                    ))
                    .with(Fun.I32ToI32.pair(
                        x -> (int) Math.round(  x >= 0 ? x : ((double) x) * .01d ),
                        x -> ( x >= 0 ? 1 : 0 )
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                            .with( "if (input >= 0) {  output = input; } else { output = input * (float)0.01; }\n" )
                            .and( "if (input >= 0) { output = (float)1; } else { output = (float)0.01; }\n" )
            )
        );
    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "relu" + expression;
        return "relu" + "(" + expression + ")";
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
        double output;
        if ( !derive ) {
            if ( input >= 0 ) output = input;
            else output = input * 0.01;
        } else {
            if ( input >= 0 ) output = 1;
            else output = 0.01;
        }
        return output;
    }


}
