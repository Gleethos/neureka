package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.CalcUtil;
import neureka.calculus.args.Arg;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Operator;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Subtraction extends AbstractOperation
{
    private static final DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                if ( d < 0 ) {
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] - t2_val[t2Idx.i()];
                } else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (d == 0) return 1;
                        else return -1;
                    };
                }
            };

    private static final DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                if ( d < 0 ) {
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.indexOfIndices( t1Idx )] - t2_val[ndc2.indexOfIndices(t2Idx)];
                } else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (d == 0) return 1;
                        else return -1;
                    };
                }
            };

    public Subtraction()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "subtract"    )
                        .setOperator(         "-"        )
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
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return ( t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] - t2_val[t2Idx.i()];
                    } else return ( t1Idx, t2Idx ) -> ( d == 0 ) ? 1.0 : -1.0;
                };
        DefaultOperatorCreator<PrimaryNDAConsumer> operationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] - t2_val[ndc2.indexOfIndices( t1Idx )];
                    } else return t1Idx -> ( d == 0 ) ? 1.0 : -1.0;
                };

        Operator operator = new Operator(JunctionUtil::forSubtractions)
                                   .setSupplyADAgentFor(
                                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                                    )
                                    .build();

        setAlgorithm(
                Operator.class,
                operator.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Operator.operate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        call.getTsrOfType( Number.class, 1 ),
                                                                        call.getTsrOfType( Number.class, 2 ),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        operationXCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Operator.operate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        call.getTsrOfType( Number.class, 1 ),
                                                                        call.getTsrOfType( Number.class, 2 ),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        operationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operator.getKernelSource() )
                                .activationSource( "output = input1 - input2;  \n" )
                                .differentiationSource(
                                        "if (d==0) {                  \n" +//drn and src2 switch:
                                        "    output = 1;              \n" +
                                        "} else {                     \n" +
                                        "    output = -1;               " +
                                        "}"
                                )
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
                                                    .pass( call.getDerivativeIndex() )
                                                    .call( gwz );
                                        }
                                )
                                .build()
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarOperatorCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return t1Idx -> t1_val[ t1Idx.i() ] - value;
                    else if ( d == 0 ) return t1Idx -> 1; else return t1Idx -> -1;
                };

        ScalarOperatorCreator<PrimaryNDAConsumer> scalarOperatorXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] - value;
                    else if ( d == 0 ) return t1Idx -> 1; else return t1Idx -> -1;
                };

        Scalarization scalarization = new Scalarization()
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                )
                .setOrchestration( (caller, call) -> CalcUtil.executeFor( caller, call, JunctionUtil::forSubtractions ) )
                .build();

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call -> {
                                    int offset = (call.getTsrOfType( Number.class, 2 ).isVirtual() || call.getTsrOfType( Number.class, 2 ).size() == 1) ? 1 : 0;
                                    double value = call.getTsrOfType( Number.class, 1+offset).value64( 0 );
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTsrOfType( Number.class, 0 ).size(),
                                                    (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                    ? ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    start, end,
                                                                    scalarOperatorXCreator.create(call.getTensors(), value, -1)
                                                            )
                                                    : ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    start, end,
                                                                    scalarOperatorCreator.create(call.getTensors(), value, -1)
                                                            )
                                            );
                                },
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( scalarization.getKernelSource() )
                                .activationSource( "output = input1 - value;\n" )
                                .differentiationSource(
                                        "if (d==0) {     \n" +//drn and src2 switch:
                                        "    output = 1;  \n" +
                                        "} else {         \n" +
                                        "    output = -1;   " +
                                        "}"
                                )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 2 ).isVirtual() || call.getTsrOfType( Number.class, 2 ).size() == 1)?1:0;
                                            int gwz = call.getTsrOfType( Number.class, 0 ).size();
                                            call.getDevice().getKernel(call)
                                                    .passAllOf(call.getTsrOfType( Number.class, 0 ))
                                                    .passAllOf(call.getTsrOfType( Number.class, 0 ))
                                                    .pass((float)call.getTsrOfType( Number.class, 1+offset).value64( 0 ))
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

        Broadcast broadcast = new Broadcast(JunctionUtil::forSubtractions)
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        {
                            Tsr<?> ctxDerivative = (Tsr<?>)call.getValOf(Arg.Derivative.class);
                            Function mul = Neureka.get().context().getFunction().mul();
                            if ( ctxDerivative != null ) {
                                return new DefaultADAgent( ctxDerivative )
                                        .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                                        .setBackward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) );
                            }
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();
                            if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                            else
                            {
                                Tsr deriv = f.derive( inputs, d );
                                return new DefaultADAgent( deriv )
                                        .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                        .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, deriv } ) );
                            }
                        }
                )
                .build();

        setAlgorithm(
                Broadcast.class,
                        broadcast
                        .setImplementationFor(
                            HostCPU.class,
                            new HostImplementation(
                                    call ->
                                            call.getDevice().getExecutor()
                                                    .threaded (
                                                            call.getTsrOfType( Number.class, 0 ).size(),
                                                            (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                                    ? ( start, end ) ->
                                                                    Broadcast.broadcast (
                                                                            call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                            call.getDerivativeIndex(), start, end,
                                                                            _creatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                    )
                                                                    : ( start, end ) ->
                                                                    Broadcast.broadcast (
                                                                            call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                            call.getDerivativeIndex(), start, end,
                                                                            _creator.create(call.getTensors(), call.getDerivativeIndex())
                                                                    )
                                                    ),
                                    3
                            )
                    ).setImplementationFor(
                            OpenCLDevice.class,
                            CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( broadcast.getKernelSource() )
                                .activationSource( "value = src1 - src2;\n" )
                                .differentiationSource( "value += handle - drain;\n" )
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
                                                    .pass( call.getDerivativeIndex() )
                                                    .call( gwz );
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
                reconstructed.append(" - ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return ( ( children[0].dependsOn(derivationIndex) ) ? "" : "-" ) +
                    Arrays.stream( children )
                    .filter( child -> child.dependsOn(derivationIndex) )
                    .map( child -> child.getDerivative(derivationIndex) )
                    .map( Object::toString )
                    .collect( Collectors.joining( " - " ) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.length; i++ ) {
                if (i == 0) {
                    derivative += src[ i ].derive( inputs, d, j );
                } else {
                    derivative -= src[ i ].derive( inputs, d, j );
                }
            }
            return derivative;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.length; i++ ) {
                if ( i == 0 ) {
                    derivative += src[ i ].derive( inputs, d );
                } else {
                    derivative -= src[ i ].derive( inputs, d );
                }
            }
            return derivative;
        }
    }



}
