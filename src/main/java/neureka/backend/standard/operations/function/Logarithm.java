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

public final class Logarithm extends AbstractOperation
{

    private final DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> Math.log(t1_val[ t1Idx.i() ]);
                else return ( t0Idx, t1Idx, t2Idx ) -> 1.0/(t1_val[ t1Idx.i() ]);
            };
    private final DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> Math.log(t1_val[inputs[ 1 ].indexOfIndices( t1Idx )]);
                else return ( t0Idx, t1Idx, t2Idx ) -> 1.0/(t1_val[inputs[ 1 ].indexOfIndices( t1Idx )]);
            };

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
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operationAlgorithm.getKernelSource() )
                                .activationSource( "output = log( input );\n" )
                                .differentiationSource( "output = 1.0 / ( input );\n" )
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
        if ( expression.startsWith("(") && expression.endsWith(")") ) return "ln" + expression;
        return "ln" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        if (children.length != 1) throw new IllegalStateException("Natual logarithm does not support more than 1 argument.");
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
