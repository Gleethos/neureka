package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Algorithm;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.Operator;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
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
        if ( d < 0 ) {
            return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] / t2_val[t2Idx.i()];
        } else {
            return ( t0Idx, t1Idx, t2Idx ) -> {
                if (d == 0) {//"    output = 1/input2;\n" +
                    return 1 / t2_val[t2Idx.i()];
                } else {
                    return -(t1_val[t2Idx.i()] / Math.pow(t2_val[t1Idx.i()], 2));
                }//"    output = -input2 /(float)pow(input1, 2.0f);\n" +
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
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.indexOfIndices( t1Idx )] / t2_val[ndc2.indexOfIndices(t2Idx)];
                } else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (d == 0) {//"    output = 1/input2;\n" +
                            return 1 / t2_val[ndc2.indexOfIndices(t2Idx)];
                        } else {
                            return -(t1_val[ndc2.indexOfIndices(t2Idx)] / Math.pow(t2_val[ndc1.indexOfIndices( t1Idx )], 2));
                        }//"    output = -input2 /(float)pow(input1, 2.0f);\n" +
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

        Algorithm.RecursiveJunctor rja = JunctionUtil::forDivisionsOrModuli;

        //_____________________
        // DEFAULT OPERATION :

        final DefaultOperatorCreator<SecondaryNDIConsumer> _operationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return ( t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] / t2_val[t2Idx.i()];
                    } else {
                        return ( t1Idx, t2Idx ) -> {
                            if (d == 0) {//"    output = 1/input2;\n" +
                                return 1 / t2_val[t2Idx.i()];
                            } else {
                                return -(t1_val[t2Idx.i()] / Math.pow(t2_val[t1Idx.i()], 2));
                            }//"    output = -input2 /(float)pow(input1, 2.0f);\n" +
                        };
                    }
                };

        final DefaultOperatorCreator<PrimaryNDAConsumer> _operationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[inputs[ 1 ].indexOfIndices( t1Idx )] / t2_val[inputs[ 2 ].indexOfIndices( t1Idx )];
                    } else {
                        return t1Idx -> {
                            if (d == 0) {//"    output = 1/input2;\n" +
                                return 1 / t2_val[inputs[ 2 ].indexOfIndices( t1Idx )];
                            } else {
                                return -(t1_val[inputs[ 2 ].indexOfIndices( t1Idx )] / Math.pow(t2_val[inputs[ 1 ].indexOfIndices( t1Idx )], 2));
                            }//"    output = -input2 /(float)pow(input1, 2.0f);\n" +
                        };
                    }
                };

        Operator operator = new Operator()
                   .setSupplyADAgentFor(
                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                )
                .setHandleRecursivelyAccordingToArity( rja )
                .build();

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
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                            ? ( start, end ) ->
                                                                Operator.operate (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    call.getTsrOfType( Number.class, 1 ),
                                                                    call.getTsrOfType( Number.class, 2 ),
                                                                    call.getDerivativeIndex(),
                                                                    start, end,
                                                                    _operationXCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                            : ( start, end ) ->
                                                                Operator.operate (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    call.getTsrOfType( Number.class, 1 ),
                                                                    call.getTsrOfType( Number.class, 2 ),
                                                                    call.getDerivativeIndex(),
                                                                    start, end,
                                                                    _operationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                    )
                    .setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTsrOfType( Number.class, offset ) )
                                            .pass( call.getTsrOfType( Number.class, offset + 1 ) )
                                            .pass( call.getTsrOfType( Number.class, offset + 2 ) )
                                            .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                operator.getKernelSource(), // kernelSource
                                "output = input1 / input2;\n",
                                "if (d==0) {\n" +
                                        "    output = 1/input2;\n" +
                                        "} else {\n" +
                                        "    output = -input2 /(float)pow(input1, 2.0f);\n" +
                                        "}",
                                this // OperationType
                        )
                    )
        );


        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast()
                                        .setCanPerformBackwardADFor( call -> true )
                                        .setCanPerformForwardADFor( call -> {
                                                Tsr<?> last = null;
                                                for ( Tsr<?> t : call.getTensors() ) {
                                                    if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                                    last = t;
                                                }
                                                return true;
                                        } )
                                        .setSupplyADAgentFor(
                                            ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                            {
                                                Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                                                Function mul = Function.Detached.MUL;
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
                                        .setHandleRecursivelyAccordingToArity( rja )
                                        .build();

        setAlgorithm(
                Broadcast.class,
                broadcast.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
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
                        new CLImplementation(
                                call -> {
                                    int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTsrOfType( Number.class, offset ) )
                                            .pass( call.getTsrOfType( Number.class, offset + 1 ) )
                                            .pass( call.getTsrOfType( Number.class, offset + 2 ) )
                                            .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                broadcast.getKernelSource(), // kernelSource
                                "value = src1 / src2;\n",
                                "if (d==0) {\n" +
                                        "    value += (1/handle) * drain;\n" +
                                        "} else {\n" +
                                        "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                                        "}",
                                this // OperationType
                        )
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
                .setHandleInsteadOfDevice( (caller, call ) -> null )
                .setHandleRecursivelyAccordingToArity( rja )
                .build();

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
                                                    (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                    ? ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    start, end,
                                                                    scalarXCreator.create(call.getTensors(), value, call.getDerivativeIndex())
                                                            )
                                                    : ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    start, end,
                                                                    scalarCreator.create(call.getTensors(), value, call.getDerivativeIndex())
                                                            )
                                            );
                                },
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = (call.getTsrOfType( Number.class, 2 ).isVirtual() || call.getTsrOfType( Number.class, 2 ).size() == 1)?1:0;
                                    int gwz = call.getTsrOfType( Number.class, 0 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTsrOfType( Number.class, 0 ))
                                            .pass(call.getTsrOfType( Number.class, 0 ))
                                            .pass((float)call.getTsrOfType( Number.class, 1+offset).value64( 0 ))
                                            .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                scalarization.getKernelSource(), // kernelSource
                                "output = input1 / value;\n",
                                "if (d==0) {\n" +
                                        "    output = 1/value;\n" +
                                        "} else {\n" +
                                        "    output = -value /(float)pow(input1, 2.0f);\n" +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //__________________________
        // RELATED OPERATION TYPES :

        new AbstractOperation(
                new OperationBuilder()
                        .setFunction(         "inv_division_left" )
                        .setOperator(         ((char) 171) + "/"  )
                        .setArity(            3          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( false      )
                        .setIsInline(         false      )
        ) {;
            @Override
            public String stringify(String[] children) {
                return null;
            }

            @Override
            public String asDerivative( Function[] children, int d ) {
                throw new IllegalStateException("Operation does not support dynamic derivation!");
            }

            @Override
            public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
            }
        };

        new AbstractOperation(
                new OperationBuilder()
                        .setFunction(         "inv_division_right" )
                        .setOperator(         "/" + ((char) 187)  )
                        .setArity(            3          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( false      )
                        .setIsInline(         false      )
        ) {;
            @Override
            public String stringify(String[] children) {
                return null;
            }

            @Override
            public String asDerivative( Function[] children, int d ) {
                throw new IllegalStateException("Operation does not support dynamic derivation!");
            }

            @Override
            public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
            }
        };

        // Convolution:

        new AbstractOperation(
                new OperationBuilder()
                        .setFunction(         "divide"           )
                        .setOperator(         "d"                )
                        .setArity(            2                  )
                        .setIsOperator(       true               )
                        .setIsIndexer(        false              )
                        .setIsDifferentiable( true               )
                        .setIsInline(         false              )
        ) {
            @Override
            public String stringify(String[] children) {
                StringBuilder reconstructed = new StringBuilder();
                for ( int i = 0; i < children.length; ++i ) {
                    reconstructed.append( children[ i ] );
                    if ( i < children.length - 1 ) {
                        reconstructed.append(" d ");
                    }
                }
                return "(" + reconstructed + ")";
            }

            @Override
            public String asDerivative( Function[] children, int d ) {
                throw new IllegalStateException("Operation does not support dynamic derivation!");
            }

            @Override
            public double calculate( double[] inputs, int j, int d, Function[] src ) {
                return 0;
            }
        }.setAlgorithm(
                Convolution.class,
                new Convolution()
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
                    ).setSupplyADAgentFor(
                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        {
                            Tsr<?> ctxDerivative = (Tsr<?>) call.getAt("derivative");
                            Function mul = Function.Detached.MUL;
                            if ( ctxDerivative != null ) {
                                return new DefaultADAgent( ctxDerivative )
                                        .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                                        .setBackward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) );
                            }
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();
                            if ( forward )
                                throw new IllegalArgumentException("Convolution of does not support forward-AD!");
                            else
                            {
                                Tsr<?> localDerivative = f.derive( inputs, d );
                                return new DefaultADAgent( localDerivative )
                                        .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, localDerivative}) )
                                        .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, localDerivative}) );
                            }
                        }
                    )
                    .setHandleInsteadOfDevice( (caller, call ) -> null )
                    .setHandleRecursivelyAccordingToArity( (call, goDeeperWith ) -> null )
                    .setInstantiateNewTensorsForExecutionIn(
                            call -> {
                                Tsr[] tsrs = call.getTensors();
                                int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                                return ExecutionCall.builder()
                                            .device( call.getDevice() )
                                            .tensors( new Tsr[]{tsrs[offset], tsrs[1+offset]} )
                                            .derivativeIndex( -1 )
                                            .operation( OperationContext.get().instance("idy") )
                                            .build();
                            }
                    )
                    .build()
                );

        new AbstractOperation(
                new OperationBuilder()
                        .setFunction(         "" )
                        .setOperator(         ((char) 171) + "d"  )
                        .setArity(            3          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true      )
                        .setIsInline(         false      )
        ) {
            @Override
            public String stringify(String[] children) {
                StringBuilder reconstructed = new StringBuilder();
                for ( int i = 0; i < children.length; ++i ) {
                    reconstructed.append( children[ i ] );
                    if ( i < children.length - 1 ) {
                        reconstructed.append(" "+((char) 171) + "d ");
                    }
                }
                return "(" + reconstructed + ")";
            }

            @Override
            public String asDerivative( Function[] children, int d ) {
                throw new IllegalStateException("Operation does not support dynamic derivation!");
            }

            @Override
            public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
            }
        };

        new AbstractOperation(
                new OperationBuilder()
                        .setFunction(         ""                 )
                        .setOperator(         "d" + ((char) 187) )
                        .setArity(            3                  )
                        .setIsOperator(       true               )
                        .setIsIndexer(        false              )
                        .setIsDifferentiable( true               )
                        .setIsInline(         false              )
        ) {
            @Override
            public String stringify(String[] children) {
                StringBuilder reconstructed = new StringBuilder();
                for ( int i = 0; i < children.length; ++i ) {
                    reconstructed.append( children[ i ] );
                    if ( i < children.length - 1 ) {
                        reconstructed.append(" d" + ((char) 187)+" ");
                    }
                }
                return "(" + reconstructed + ")";
            }

            @Override
            public String asDerivative( Function[] children, int d ) {
                throw new IllegalStateException("Operation does not support dynamic derivation!");
            }

            @Override
            public double calculate( double[] inputs, int j, int d, Function[] src ) {
                return src[ 0 ].call( inputs, j );
            }
        };

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
    public String asDerivative( Function[] children, int d ) {
        return _asDerivative( children, d, children.length - 1 );
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
