package neureka.backend.standard.operations.function;

import neureka.Neureka;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

public final class ReLU extends AbstractOperation
{

    private DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                if ( d < 0 ) {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (t1_val[ t1Idx.i() ]>=0) return t1_val[ t1Idx.i() ];
                        else return t1_val[ t1Idx.i() ]*0.01;
                    };
                } else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (t1_val[ t1Idx.i() ]>=0) return 1;
                        else return 0.01;
                    };
                }
            };

    private DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                if ( d < 0 ) {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (t1_val[inputs[ 1 ].indexOfIndices( t1Idx )]>=0) return t1_val[inputs[ 1 ].indexOfIndices( t1Idx )];
                        else return t1_val[inputs[ 1 ].indexOfIndices( t1Idx )]*0.01;
                    };
                } else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (t1_val[inputs[ 1 ].indexOfIndices( t1Idx )]>=0) return 1;
                        else return 0.01;
                    };
                }
            };

    public ReLU()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "relu"    )
                        .setOperator(         "relu"    )
                        .setArity(            1        )
                        .setIsOperator(       false    )
                        .setIsIndexer(        false    )
                        .setIsDifferentiable( true     )
                        .setIsInline(         false    )
        );

        Activation operationAlgorithm = new Activation()
            .setSupplyADAgentFor(
                ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
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
                                                                        _creatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Activation.activate (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ),
                                                                        start, end,
                                                                        _creator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operationAlgorithm.getKernelSource() )
                                .activationSource( "if (input >= 0) {  output = input; } else { output = input * (float)0.01; }\n" )
                                .differentiationSource( "if (input >= 0) { output = (float)1; } else { output = (float)0.01; }\n" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                            call.getDevice().getKernel(call)
                                                    .pass( call.getTsrOfType( Number.class, offset ) )
                                                    .pass( call.getTsrOfType( Number.class, offset + 1 ) )
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
