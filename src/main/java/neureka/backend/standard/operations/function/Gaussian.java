package neureka.backend.standard.operations.function;

import neureka.Tsr;
import neureka.backend.standard.algorithms.Fun;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class Gaussian extends AbstractOperation
{

    public Gaussian()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "gaus"    )
                        .setOperator(         "gaus"    )
                        .setArity(            1         )
                        .setIsOperator(       false     )
                        .setIsIndexer(        false     )
                        .setIsDifferentiable( true      )
                        .setIsInline(         false     )
        );

        Activation operationAlgorithm = new Activation()
            .setCanPerformBackwardADFor( call -> true )
            .setCanPerformForwardADFor(
                    call -> {
                        Tsr<?> last = null;
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                    }
            )
            .setSupplyADAgentFor( getDefaultAlgorithm() )
        .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
        .setCallPreparation(
            call -> {
                Tsr<?>[] inputs = call.getTensors();
                Device device = call.getDevice();
                if ( inputs[ 0 ] == null ) // Creating a new tensor:
                {
                    int[] shp = inputs[ 1 ].getNDConf().shape();
                Tsr<?> output = Tsr.of( shp, 0.0 ).getMutate().setIsIntermediate( true );
                output.setIsVirtual( false );
                try {
                    device.store( output );
                } catch( Exception e ) {
                    e.printStackTrace();
                }
                inputs[ 0 ] = output;
                }
                return call;
            }
        )
        .buildFunAlgorithm();

        setAlgorithm(
            Activation.class,
            operationAlgorithm.setImplementationFor(
                CPU.class,
                CPUImplementation
                    .withArity(3)
                    .andImplementation(
                        Activation.implementationForCPU()
                                .with(Fun.F64ToF64.pair(
                                    x -> Math.pow(Math.E, -Math.pow(x, 2)),
                                    x -> -2 * x * Math.pow(Math.E, -Math.pow(x, 2))
                                ))
                                .with(Fun.F32ToF32.pair(
                                    x -> (float) Math.pow(Math.E, -Math.pow(x, 2)),
                                    x -> (float) (-2 * x * Math.pow(Math.E, -Math.pow(x, 2)))
                                ))
                                .get()
                    )
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation
                    .compiler()
                    .arity( 3 )
                    .kernelSource( operationAlgorithm.getKernelSource() )
                    .activationSource(
                        "output =\n" +
                        "    (float)pow(\n" +
                        "        (float)M_E,\n" +
                        "        -(float)pow(\n" +
                        "            (float)input,\n" +
                        "            (float)2\n" +
                        "        )\n" +
                        "    );\n"
                    )
                    .differentiationSource(
                        "output = 1 / (1 + (float)pow((float)M_E, -input));\n"
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
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "gaus" + expression;
        return "gaus" + "(" + expression + ")";
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
