package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Operator;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Addition extends AbstractOperation {

    private static final DefaultOperatorCreator<TertiaryNDIConsumer> _broadcastCreator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].getValueAs( double[].class );
                double[] t2_val = inputs[ 2 ].getValueAs( double[].class );
                // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] + t2_val[t2Idx.i()];
            };

    private static final DefaultOperatorCreator<TertiaryNDAConsumer> _broadcastCreatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].getValueAs( double[].class );
                double[] t2_val = inputs[ 2 ].getValueAs( double[].class );
                NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.indexOfIndices( t1Idx )] + t2_val[ndc2.indexOfIndices(t2Idx)];
            };


    private final Broadcast _broadcast = new Broadcast((executionCall, executor) -> null)
                                                    .setCanPerformBackwardADFor( call -> true )
                                                    .setSupplyADAgentFor(
                                                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                                        {
                                                            Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                                                            assert ctxDerivative == null;
                                                            Tsr<?>[] inputs = call.getTensors();
                                                            int d = call.getDerivativeIndex();
                                                            if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                                                            else
                                                            {
                                                                Tsr<?> derivative = inputs[(d==0?1:0)];
                                                                Tsr<?> toBeDerived = inputs[d];
                                                                Device device = call.getDeviceFor(Number.class);
                                                                return ADAgent.of( derivative )
                                                                                .setBackward(
                                                                                    ( node, backwardError ) ->
                                                                                        this.getAlgorithm( Broadcast.class )
                                                                                             .getImplementationFor( device )
                                                                                             .runAndGetFirstTensor(
                                                                                                     ExecutionCall.of(
                                                                                                             JunctionUtil.newTsrLike(toBeDerived, 0).setIsVirtual(false),
                                                                                                             JunctionUtil.newTsrLike(inputs[(d==0?1:0)], 0),
                                                                                                             backwardError
                                                                                                     )
                                                                                                     .andArgs( Arg.DerivIdx.of(d) )
                                                                                                     .running( this )
                                                                                                     .on( device )
                                                                                             )
                                                                                );
                                                            }
                                                        }
                                                    )
                                                    .buildFunAlgorithm();

    public Addition()
    {
        super (
                new OperationBuilder()
                        .setFunction(         "add"      )
                        .setOperator(         "+"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<SecondaryNDIConsumer> operationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].getValueAs( double[].class );
                    double[] t2_val = inputs[ 2 ].getValueAs( double[].class );
                    if ( d < 0 ) return ( t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] + t2_val[t2Idx.i()];
                    else return ( t1Idx, t2Idx ) -> 1.0;
                };

        DefaultOperatorCreator<PrimaryNDAConsumer> operationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].getValueAs( double[].class );
                    double[] t2_val = inputs[ 2 ].getValueAs( double[].class );
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] + t2_val[ndc2.indexOfIndices( t1Idx )];
                    else return t1Idx -> 1.0;
                };

        Operator operator = new Operator(JunctionUtil::forAdditions)
                                    .setSupplyADAgentFor( getDefaultAlgorithm() )
                                    .buildFunAlgorithm();

        setAlgorithm(
                operator
                        .setImplementationFor(
                            CPU.class,
                                CPUImplementation
                                    .withArity(3)
                                    .andImplementation(
                                        call ->
                                                call.getDevice()
                                                        .getExecutor()
                                                        .threaded (
                                                                call.getTsrOfType( Number.class, 0 ).size(),
                                                                (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                                ? ( start, end ) ->
                                                                        Operator.operate (
                                                                                call.getTsrOfType( Number.class, 0 ),
                                                                                call.getTsrOfType( Number.class, 1 ),
                                                                                call.getTsrOfType( Number.class, 2 ),
                                                                                call.getValOf( Arg.DerivIdx.class ),
                                                                                start, end,
                                                                                operationXCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                        )
                                                                : ( start, end ) ->
                                                                        Operator.operate (
                                                                                call.getTsrOfType( Number.class, 0 ),
                                                                                call.getTsrOfType( Number.class, 1 ),
                                                                                call.getTsrOfType( Number.class, 2 ),
                                                                                call.getValOf( Arg.DerivIdx.class ),
                                                                                start, end,
                                                                                operationCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                        )
                                                        )
                                    )
                        )
                        .setImplementationFor(
                            OpenCLDevice.class,
                            CLImplementation.compiler()
                                    .arity( 3 )
                                    .kernelSource( operator.getKernelSource() )
                                    .activationSource( "output = input1 + input2;\n" )
                                    .differentiationSource( "output = 1;\n" )
                                    .kernelPostfix( this.getFunction() )
                                    .execution(
                                            call -> {
                                                int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                                int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                                call.getDevice()
                                                        .getKernel(call)
                                                        .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                                        .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                                        .passAllOf( call.getTsrOfType( Number.class, offset + 2 ) )
                                                        .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                        .pass( call.getDerivativeIndex() )
                                                        .call( gwz );
                                            }
                                    )
                                    .build()
                )
        );

        //________________
        // BROADCASTING :

        setAlgorithm(
                _broadcast
                .setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
                                call -> {
                                    call.getDevice()
                                            .getExecutor()
                                            .threaded(
                                                    call.getTsrOfType(Number.class, 0).size(),
                                                    (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                            ? (start, end) ->
                                                            Broadcast.broadcast(
                                                                    call.getTsrOfType(Number.class, 0), call.getTsrOfType(Number.class, 1), call.getTsrOfType(Number.class, 2),
                                                                    call.getValOf(Arg.DerivIdx.class), start, end,
                                                                    _broadcastCreatorX.create(call.getTensors(), call.getValOf(Arg.DerivIdx.class))
                                                            )
                                                            : (start, end) ->
                                                            Broadcast.broadcast(
                                                                    call.getTsrOfType(Number.class, 0), call.getTsrOfType(Number.class, 1), call.getTsrOfType(Number.class, 2),
                                                                    call.getValOf(Arg.DerivIdx.class), start, end,
                                                                    _broadcastCreator.create(call.getTensors(), call.getValOf(Arg.DerivIdx.class))
                                                            )
                                            );
                                }
                            )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( _broadcast.getKernelSource() )
                                .activationSource( "value += src1 + src2;\n" )
                                .differentiationSource( "value += src1 + src2;\n" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                            call.getDevice().getKernel(call)
                                                    .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                                    .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                                    .passAllOf( call.getTsrOfType( Number.class, offset + 2 ) )
                                                    .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                                    .call( gwz );
                                        }
                                )
                                .build()
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        Scalarization scalarization = new Scalarization()
                                            .setSupplyADAgentFor( getDefaultAlgorithm() )
                                            .setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, JunctionUtil::forAdditions ) )
                                            .buildFunAlgorithm();

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].getValueAs( double[].class );
                    if ( d < 0 ) return t1Idx -> t1_val[ t1Idx.i() ] + value;
                    else
                        return t1Idx -> 1;
                };

        ScalarOperatorCreator<PrimaryNDAConsumer> scalarXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].getValueAs( double[].class );
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] + value;
                    else
                        return t1Idx -> 1;
                };

        setAlgorithm(
                scalarization.setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
                                call -> {
                                    assert call.getTensors().length == 3;
                                    if ( call.getDerivativeIndex() == 0 )
                                        call.getTensors()[0] = Tsr.of( call.getTensors()[1].shape(), 1d );
                                    else if ( call.getDerivativeIndex() == 1 )
                                        call.getTensors()[0] = Tsr.of( call.getTensors()[2].shape(), 1d );
                                    else {
                                        double value = call.getTsrOfType(Number.class, 2).getValueAt(0).doubleValue();
                                        call.getDevice()
                                                .getExecutor()
                                                .threaded(
                                                        call.getTsrOfType(Number.class, 0).size(),
                                                        (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                                ? (start, end) ->
                                                                Scalarization.scalarize(
                                                                        call.getTsrOfType(Number.class, 0),
                                                                        start, end,
                                                                        scalarXCreator.create(call.getTensors(), value, -1)
                                                                )
                                                                : (start, end) ->
                                                                Scalarization.scalarize(
                                                                        call.getTsrOfType(Number.class, 0),
                                                                        start, end,
                                                                        scalarCreator.create(call.getTensors(), value, -1)
                                                                )
                                                );
                                        }
                                    }
                                )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( scalarization.getKernelSource() )
                                .activationSource( "output = input1 + value;\n" )
                                .differentiationSource( "output = 1;\n" )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            assert call.getTensors().length == 3;
                                            if ( call.getDerivativeIndex() == 0 )
                                                call.getTensors()[0] = Tsr.of( call.getTensors()[1].shape(), 1d );
                                            else if ( call.getDerivativeIndex() == 1 )
                                                call.getTensors()[0] = Tsr.of( call.getTensors()[2].shape(), 1d );
                                            else {
                                                int gwz = call.getTsrOfType(Number.class, 0).size();
                                                float value = call.getTsrOfType(Number.class, 2).getValueAt(0).floatValue();
                                                call.getDevice().getKernel(call)
                                                        .passAllOf(call.getTsrOfType(Number.class, 0))
                                                        .passAllOf(call.getTsrOfType(Number.class, 1))
                                                        .pass(value)
                                                        .pass(call.getTsrOfType(Number.class, 0).rank())
                                                        .pass(call.getValOf(Arg.DerivIdx.class))
                                                        .call(gwz);
                                            }
                                        }
                                )
                                .build()
                )
        );
    }

    @Contract(pure = true)
    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" + ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        String s =  Arrays.stream( children )
                .filter( child -> child.dependsOn(derivationIndex) )
                .map( child -> child.getDerivative(derivationIndex) )
                .map( Object::toString )
                .collect( Collectors.joining( " + " ) );
        return s;
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( Function function : src )
                derivative += function.derive(inputs, d, j);

            return derivative;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( Function function : src ) {
                derivative += function.derive( inputs, d );
            }
            return derivative;
        }
    }




}
