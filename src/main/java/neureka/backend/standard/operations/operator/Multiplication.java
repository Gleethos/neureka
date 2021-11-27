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


public class Multiplication extends AbstractOperation
{

    private static final DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] * t2_val[t2Idx.i()];
                else return ( t0Idx, t1Idx, t2Idx ) -> (d == 0) ? t2_val[t2Idx.i()] : t1_val[ t1Idx.i() ];
            };

    private static final DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                if ( d < 0 ) {
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.indexOfIndices( t1Idx )] * t2_val[ndc2.indexOfIndices(t2Idx)];
                } else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if ( d == 0 ) return t2_val[ndc2.indexOfIndices(t2Idx)];
                        else return t1_val[ndc1.indexOfIndices( t1Idx )];
                    };
                }
            };

    public static DefaultOperatorCreator<TertiaryNDIConsumer> xBCCreator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] * t2_val[t2Idx.i()];
            };

    public static DefaultOperatorCreator<TertiaryNDAConsumer> xBCCreatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.indexOfIndices( t1Idx )] * t2_val[ndc2.indexOfIndices(t2Idx)];
            };

    public Multiplication()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "multiply"    )
                        .setOperator(         "*"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<SecondaryNDIConsumer> defaultOperatorcreator =
                ( inputs, d ) -> {
                    inputs[ 1 ].setIsVirtual( false );
                    inputs[ 2 ].setIsVirtual( false );
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return ( t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] * t2_val[t2Idx.i()];
                    } else {
                        return ( t1Idx, t2Idx ) -> {
                            if ( d == 0 ) return t2_val[t2Idx.i()];
                            else return t1_val[ t1Idx.i() ];
                        };
                    }
                };


        DefaultOperatorCreator<PrimaryNDAConsumer> defaultOperatorXcreator =
                ( inputs, d ) -> {
                    inputs[ 1 ].setIsVirtual( false );
                    inputs[ 2 ].setIsVirtual( false );
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] * t2_val[ndc2.indexOfIndices( t1Idx )];
                    } else {
                        return t1Idx -> {
                            if ( d == 0 ) return t2_val[ndc2.indexOfIndices( t1Idx )];
                            else return t1_val[ndc1.indexOfIndices( t1Idx )];
                        };
                    }
                };

        Operator operator = new Operator(JunctionUtil::forMultiplications)
                                   .setSupplyADAgentFor(
                                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                                    )
                                    .buildFunAlgorithm();

        setAlgorithm(
                Operator.class,
                operator.setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
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
                                                                        call.getValOf( Arg.DerivIdx.class ),
                                                                        start, end,
                                                                        defaultOperatorXcreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                        : ( start, end ) ->
                                                                Operator.operate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        call.getTsrOfType( Number.class, 1 ),
                                                                        call.getTsrOfType( Number.class, 2 ),
                                                                        call.getValOf( Arg.DerivIdx.class ),
                                                                        start, end,
                                                                        defaultOperatorcreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                )
                            )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operator.getKernelSource() )
                                .activationSource( "output = input1 * input2;\n" )
                                .differentiationSource( "if ( d==0 ) {output = input2;}else{output = input1;}\n" )
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


        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast( JunctionUtil::forMultiplications )
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> false )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        Function mul = Neureka.get().backend().getFunction().mul();
                        if ( ctxDerivative != null ) {
                            return ADAgent.of( ctxDerivative )
                                            .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) )
                                            .setBackward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) );
                        }
                        Tsr<?>[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr<?> derivative = f.executeDerive( inputs, d );
                            return ADAgent.of( derivative )
                                    .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                    .setBackward( (node, backwardError ) -> mul.execute( backwardError, derivative ) );
                        }
                    }
                )
                .buildFunAlgorithm();

        setAlgorithm(
                Broadcast.class,
                broadcast
                    .setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
                                call ->
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTsrOfType( Number.class, 0 ).size(),
                                                    (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                    ? ( start, end ) ->
                                                            Broadcast.broadcast (
                                                                call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                call.getValOf( Arg.DerivIdx.class ), start, end,
                                                                _creatorX.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                            )
                                                    : ( start, end ) ->
                                                            Broadcast.broadcast (
                                                                call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                call.getValOf( Arg.DerivIdx.class ), start, end,
                                                                _creator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                            )
                                            )
                            )
                    )
                    .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                            .arity( 3 )
                            .kernelSource( broadcast.getKernelSource() )
                            .activationSource( "value = src1 * src2;\n" )
                            .differentiationSource( "value += handle * drain;\n" )
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

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarOperatorCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return t1Idx -> t1_val[ t1Idx.i() ] * value;
                    else {
                        if ( d == 0 ) return t1Idx -> value;
                        else return t1Idx -> t1_val[ t1Idx.i() ];
                    }
                };

        ScalarOperatorCreator<PrimaryNDAConsumer> scalarOperatorXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] * value;
                    else {
                        if ( d == 0 ) return t1Idx -> value;
                        else return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )];
                    }
                };

        Scalarization scalarization = new Scalarization()
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        Function mul = Neureka.get().backend().getFunction().mul();
                        if ( ctxDerivative != null ) {
                            return ADAgent.of( ctxDerivative )
                                    .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) )
                                    .setBackward( null );
                        }
                        Tsr<?>[] inputs = call.getTensors();
                        int d = call.getValOf( Arg.DerivIdx.class );
                        if ( forward )
                        {
                            Tsr<?> derivative = f.executeDerive( inputs, d );
                            return ADAgent.of( derivative )
                                    .setForward( (t, graphDerivative ) -> mul.execute( graphDerivative, derivative ) )
                                    .setBackward( null );
                        }
                        else
                        {
                            Tsr<?> derivative = f.executeDerive( inputs, d );
                            return ADAgent.of( derivative )
                                    .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                    .setBackward( (node, backwardError ) -> mul.execute( backwardError, derivative ) );
                        }
                    }
                )
                .setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, JunctionUtil::forMultiplications ) )
                .buildFunAlgorithm();

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
                                call -> {
                                    double value = call.getTsrOfType( Number.class, 0 ).value64( 2 );
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
                                }
                            )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( scalarization.getKernelSource() )
                                .activationSource( "output = input1 * value;\n" )
                                .differentiationSource( "if ( d == 0 ) {output = value;}else{output = input1;}\n" )
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
                                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
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
                reconstructed.append(" * ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return Arrays.stream( children )
                .filter( child -> child.dependsOn(derivationIndex) )
                .map( child -> {
                            String derivative = child.getDerivative(derivationIndex).toString();
                            return ( (derivative.equals("1.0") ) ? "" : " * " ) +
                                    Arrays.stream( children )
                                            .filter( inner -> inner != child )
                                            .map( Object::toString )
                                            .collect( Collectors.joining( " * " ) );
                        }
                )
                .map( Object::toString )
                .collect( Collectors.joining( " + " ) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result *= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs, j );
            ud = src[ 0 ].derive( inputs, d, j );

            for ( int ji = 1; ji < src.length; ji++ ) {
                v = src[ ji ].call( inputs, j );
                vd = src[ ji ].derive( inputs, d, j );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result *= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs );
            ud = src[ 0 ].derive( inputs, d );
            for ( int j = 1; j < src.length; j++ ) {
                v = src[ j ].call( inputs );
                vd = src[ j ].derive( inputs, d );

                ud = u * vd + v * ud;
                u *= v; // ...this step can be avoided (TODO optimize)
            }
            return ud;
        }
    }




}
