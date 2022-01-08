package neureka.backend.standard.operations.function;

import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Fun;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
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
            .setSupplyADAgentFor(
                ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
            )
            .buildFunAlgorithm();


        setAlgorithm(
                Activation.class,
                operationAlgorithm
                        .setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
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
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operationAlgorithm.getKernelSource() )
                                .activationSource(
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
                                .differentiationSource(
                                        "output =\n" +
                                                "    1 /\n" +
                                                "        (1 + (float) pow(\n" +
                                                "                (float)M_E,\n" +
                                                "                (float)input\n" +
                                                "            )\n" +
                                                "        );\n"
                                )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                            call.getDevice().getKernel(call)
                                                    .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                                    .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                                    .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                                    .call( gwz );
                                        }
                                )
                                .build()
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
