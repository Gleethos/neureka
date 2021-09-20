package neureka.backend.standard.operations.function;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
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

        DefaultOperatorCreator<TertiaryNDIConsumer> activationCreator =
                ( inputs, d ) ->
                {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> Math.pow(Math.E, -Math.pow(t1_val[ t1Idx.i() ], 2));
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            double input = t1_val[ t1Idx.i() ];
                            return -2 * input * Math.pow(Math.E, -Math.pow(input, 2));
                        };

                    }
                };

        DefaultOperatorCreator<TertiaryNDAConsumer> activationXCreator =
                ( inputs, d ) ->
                {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> Math.pow(Math.E, -Math.pow(t1_val[inputs[ 1 ].indexOfIndices( t1Idx )], 2));
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            double input = t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                            return -2 * input * Math.pow(Math.E, -Math.pow(input, 2));
                        };

                    }
                };

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
            .setSupplyADAgentFor(
                ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
            )
        .setOrchestration( CalcUtil::defaultRecursiveExecution)
        .setInstantiateNewTensorsForExecutionIn(
                call -> {
                        Tsr[] tsrs = call.getTensors();
                        Device device = call.getDevice();
                        if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                        {
                            int[] shp = tsrs[ 1 ].getNDConf().shape();
                        Tsr output = Tsr.of( shp, 0.0 );
                        output.setIsVirtual( false );
                        try {
                            device.store(output);
                        } catch( Exception e ) {
                            e.printStackTrace();
                        }
                        tsrs[ 0 ] = output;
                        }
                        return call;
                    }
            )
            .build();

        setAlgorithm(
                Activation.class,
                operationAlgorithm.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call  ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        start, end,
                                                                        activationXCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ),
                                                                        start, end,
                                                                        activationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
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
                                                    .pass( call.getDerivativeIndex() )
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
