package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Fun;
import neureka.backend.standard.implementations.CPUImplementation;
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
                    .setFunction(         "softplus"    )
                    .setOperator(         "softplus"    )
                    .setArity(            1             )
                    .setIsOperator(       false         )
                    .setIsIndexer(        false         )
                    .setIsDifferentiable( true          )
                    .setIsInline(         false         )
        );

        Activation operationAlgorithm = new Activation()
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .buildFunAlgorithm();


        setAlgorithm(
            Activation.class,
            operationAlgorithm
                .setImplementationFor(
                CPU.class,
                Activation.implementationForCPU()
                    .with(Fun.F64ToF64.pair(
                            x -> Math.log(1d + Math.pow(Math.E, x)),
                            x -> 1d / (1d + Math.pow(Math.E, -x))
                        )
                    )
                    .with(Fun.F32ToF32.pair(
                            x -> (float) Math.log(1 + Math.pow(Math.E, x)),
                            x -> (float) (1f / (1f + Math.pow(Math.E, -x)))
                    )).get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                    Activation.implementationForGPU( this.getFunction() )
                            .with(
                                    "output = \n" +
                                            "   (\n" +
                                            "        (float) log(\n" +
                                            "            1+pow(\n" +
                                            "                (float)\n" +
                                            "                M_E,\n" +
                                            "                (float)\n" +
                                            "                input\n" +
                                            "            )\n" +
                                            "        )\n" +
                                            "    );"
                            )
                            .and(
                                    "output =\n" +
                                            "    1 /\n" +
                                            "        (1 + (float) pow(\n" +
                                            "                (float)M_E,\n" +
                                            "                (float)input\n" +
                                            "            )\n" +
                                            "        );\n"
                            )
            )
        );
    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "softplus" + expression;
        return "softplus" + "(" + expression + ")";
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
        if ( !derive ) return Math.log(1 + Math.pow(Math.E, input));
        else return Sigmoid.calculate(input, false);
    }



}
