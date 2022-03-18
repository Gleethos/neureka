package neureka.backend.standard.operations.function;

import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Sinus extends AbstractOperation
{
    public Sinus()
    {
        super(
            new OperationBuilder()
                .setIdentifier(         "sin"    )
                .setOperator(         "sin"    )
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
                        x -> Math.sin(x),
                        x -> Math.cos(x)
                    ))
                    .with(Fun.F32ToF32.pair(
                        x -> (float) Math.sin(x),
                        x -> (float) Math.cos(x)
                    ))
                    .with(Fun.I32ToI32.pair(
                        x -> (int) Math.round(Math.sin(x)),
                        x -> (int) Math.round(Math.cos(x))
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                            .with( "output = sin( input );\n" )
                            .and( "output = cos( input );\n" )
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
        if ( !derive ) return Math.sin( input );
        else return Math.cos( input );
    }




}
