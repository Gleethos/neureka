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
import neureka.backend.standard.implementations.HostImplementation;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;


public class Division extends AbstractOperation
{
    private static final DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
    ( inputs, d ) -> {
        double[] t1_val = inputs[ 1 ].value64();
        double[] t2_val = inputs[ 2 ].value64();
        if ( d < 0 )
            return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] / t2_val[t2Idx.i()];
        else {
            return ( t0Idx, t1Idx, t2Idx ) -> {
                if ( d == 0 )
                    return 1 / t2_val[t2Idx.i()];
                else
                    return -(t1_val[t2Idx.i()] / Math.pow(t2_val[t1Idx.i()], 2));
            };
        }
    };

    private static final DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                if ( d < 0 )
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.indexOfIndices( t1Idx )] / t2_val[ndc2.indexOfIndices(t2Idx)];
                else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if ( d == 0 )
                            return 1 / t2_val[ndc2.indexOfIndices(t2Idx)];
                        else
                            return -(t1_val[ndc2.indexOfIndices(t2Idx)] / Math.pow(t2_val[ndc1.indexOfIndices( t1Idx )], 2));
                    };
                }
            };

    public Division()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "divide"   )
                        .setOperator(         "/"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        final DefaultOperatorCreator<SecondaryNDIConsumer> _operationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 )
                        return ( t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] / t2_val[t2Idx.i()];
                    else {
                        return ( t1Idx, t2Idx ) -> {
                            if ( d == 0 )
                                return 1 / t2_val[t2Idx.i()];
                            else
                                return -(t1_val[t2Idx.i()] / Math.pow(t2_val[t1Idx.i()], 2));
                        };
                    }
                };

        final DefaultOperatorCreator<PrimaryNDAConsumer> _operationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 )
                        return t1Idx -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )] / t2_val[inputs[ 2 ].indexOfIndices( t1Idx )];
                    else {
                        return t1Idx -> {
                            if ( d == 0 )
                                return 1 / t2_val[inputs[ 2 ].indexOfIndices( t1Idx )];
                            else
                                return -(t1_val[inputs[ 2 ].indexOfIndices( t1Idx )] / Math.pow(t2_val[inputs[ 1 ].indexOfIndices( t1Idx )], 2));
                        };
                    }
                };

        Operator operator = new Operator(JunctionUtil::forDivisionsOrModuli)
                                   .setSupplyADAgentFor(
                                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                                    )
                                    .buildFunAlgorithm();

        setAlgorithm(
                Operator.class,
                operator
                    .setImplementationFor(
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
                                                                    call.getValOf( Arg.DerivIdx.class ),
                                                                    start, end,
                                                                    _operationXCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                            : ( start, end ) ->
                                                                Operator.operate (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    call.getTsrOfType( Number.class, 1 ),
                                                                    call.getTsrOfType( Number.class, 2 ),
                                                                    call.getValOf( Arg.DerivIdx.class ),
                                                                    start, end,
                                                                    _operationCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                ),
                                3
                        )
                    )
                    .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operator.getKernelSource() )
                                .activationSource( "output = input1 / input2;\n" )
                                .differentiationSource(
                                    "    if (d==0) {                                        \n" +
                                    "        output = 1 / input2;                           \n" +
                                    "    } else {                                           \n" +
                                    "        output = -input2 / (float)pow(input1, 2.0f);   \n" +
                                    "    }                                                  \n"
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


        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast( JunctionUtil::forDivisionsOrModuli )
                                        .setCanPerformBackwardADFor( call -> true )
                                        .setCanPerformForwardADFor( call -> {
                                                Tsr<?> last = null;
                                                for ( Tsr<?> t : call.getTensors() ) {
                                                    if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                                    last = t;
                                                }
                                                return true;
                                        })
                                        .setSupplyADAgentFor(
                                            ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                            {
                                                Tsr ctxDerivative = (Tsr)call.getValOf(Arg.Derivative.class);
                                                Function mul = Neureka.get().context().getFunction().mul();
                                                if ( ctxDerivative != null ) {
                                                    return ADAgent.of( ctxDerivative )
                                                                    .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                                                                    .setBackward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) );
                                                }
                                                Tsr[] inputs = call.getTensors();
                                                int d = call.getDerivativeIndex();
                                                if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                                                else
                                                {
                                                    Tsr deriv = f.derive( inputs, d );
                                                    return ADAgent.of( deriv )
                                                            .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                                            .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, deriv } ) );
                                                }
                                            }
                                        )
                                        .buildFunAlgorithm();

        setAlgorithm(
                Broadcast.class,
                broadcast.setImplementationFor(
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
                                                                        call.getValOf( Arg.DerivIdx.class ), start, end,
                                                                        _creatorX.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                        : ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                        call.getValOf( Arg.DerivIdx.class ), start, end,
                                                                        _creator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                ),
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( broadcast.getKernelSource() )
                                .activationSource( "value = src1 / src2;\n" )
                                .differentiationSource(
                                    "    if (d==0) {                                                         \n" +
                                    "        value += (1/handle) * drain;                                    \n" +
                                    "    } else {                                                            \n" +
                                    "        value += (-(handle /(float)pow(target, (float)2)) ) * drain;    \n" +
                                    "    }                                                                   \n"
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
                                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                                    .call( gwz );
                                        }
                                )
                                .build()
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[ t1Idx.i() ] / value;
                    } else {
                        if (d == 0) return t1Idx -> 1 / value;
                        else return t1Idx -> -value / Math.pow(t1_val[ t1Idx.i() ], 2);
                    }
                };

        ScalarOperatorCreator<PrimaryNDAConsumer> scalarXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] / value;
                    } else {
                        if (d == 0) return t1Idx -> 1 / value;
                        else return t1Idx -> -value / Math.pow(t1_val[ndc1.indexOfIndices( t1Idx )], 2);
                    }
                };

        Scalarization scalarization = new Scalarization()
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                )
                .setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, JunctionUtil::forDivisionsOrModuli ) )
                .buildFunAlgorithm();

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
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
                                                                    scalarXCreator.create(call.getTensors(), value, call.getValOf( Arg.DerivIdx.class ))
                                                            )
                                                    : ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    start, end,
                                                                    scalarCreator.create(call.getTensors(), value, call.getValOf( Arg.DerivIdx.class ))
                                                            )
                                            );
                                },
                                3
                        )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation
                                .compiler()
                                .arity( 3 )
                                .kernelSource( scalarization.getKernelSource() )
                                .activationSource( "output = input1 / value;\n" )
                                .differentiationSource(
                                    "    if (d==0) {                                       \n" +
                                    "        output = 1/value;                             \n" +
                                    "    } else {                                          \n" +
                                    "        output = -value /(float)pow(input1, 2.0f);    \n" +
                                    "    }                                                 \n"
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
                reconstructed.append(" / ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return _asDerivative( children, derivationIndex, children.length - 1 );
    }

    private String _asDerivative( Function[] children, int d, int index ) {
        if ( d >= 0 ) {
            if ( index <= 0 ) return children[ 0 ].getDerivative( d ).toString();
            else {
                String first = ( children[ index - 1 ].dependsOn( d ) )
                        ? "(" + _asDerivative( children, d, index - 1 )+ " / " + children[ index ]  + " )"
                        : "";

                if ( !children[ index ].dependsOn(d) ) return first;
                String s = children[ index - 1 ].toString();
                if ( s.equals("0.0") ) return first;

                return first +
                        " - ((" + // The second expression is the inner derivative (current index)! (inner times outer...)
                            s + " * " + children[ index ].getDerivative(d) +
                        ") / ( "
                            + children[ index ] + "^2 " +
                        ") )";
            }
        } else {
            if ( index <= 0 ) return children[ 0 ].toString();
            else
                return _asDerivative( children, -1, index - 1 ) + " / " + children[ index ].toString();
        }
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result /= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs, j );
            ud = src[ 0 ].derive( inputs, d, j );
            for ( int i = 0; i < src.length - 1; i++ ) {
                v = src[ i + 1 ].call( inputs, j );
                vd = src[ i + 1 ].derive( inputs, d, j );
                ud = (ud * v - u * vd) / Math.pow(v, 2);
                u /= v;
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
                result /= current;
            }
            return result;
        } else {
            double derivative = 0;
            double tempVar = src[ 0 ].call( inputs );
            derivative = src[ 0 ].derive( inputs, d );

            for ( int i = 0; i < src.length - 1; i++ ) {
                double u, ud, v, vd;
                v = src[ i + 1 ].call( inputs );
                vd = src[ i + 1 ].derive( inputs, d );
                u = tempVar;
                ud = derivative;
                derivative = ( ud * v - u * vd ) / Math.pow(v, 2);
                tempVar /= v;
            }
            return derivative;
        }
    }




}
