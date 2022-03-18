package neureka.backend.standard.operations.function;

import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Sigmoid extends AbstractOperation
{
    public Sigmoid()
    {
        super(
            new OperationBuilder()
                .setFunction(         "sig"    )
                .setOperator(         "sig"    )
                .setArity(            1        )
                .setIsOperator(       false    )
                .setIsIndexer(        false    )
                .setIsDifferentiable( true     )
                .setIsInline(         false    )
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
                        x -> calculate( x, false ),
                        x -> calculate( x, true )
                    ))
                    .with(Fun.F32ToF32.pair(
                        x -> (float) calculate( x, false ),
                        x -> (float) calculate( x, true )
                    ))
                    .with(Fun.I32ToI32.pair(
                        x -> (int) Math.round(calculate( x, false )),
                        x -> (int) Math.round(calculate( x, true  ))
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                            .with( "output = 1 / (1 + (float)pow((float)M_E, -input));\n" )
                            .and( "output = input * (1 - input);\n" )
            )
        );
    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "sig" + expression;
        return "sig" + "(" + expression + ")";
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
    public static double calculate( double input, boolean derive ) {
        if ( !derive ) return 1 / ( 1 + Math.exp( -input ) );
        else {
            double sig = calculate(input, false);
            return sig * ( 1 - sig );
        }
    }


}




