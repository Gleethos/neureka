package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.Algorithm;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.Operation;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.api.operations.OperationFactory;
import neureka.backend.standard.algorithms.Broadcast;
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

public class Modulo extends AbstractOperation {

    public Modulo()
    {
        super(
                new OperationFactory()
                        .setFunction(         "modulo"    )
                        .setOperator(         "%"         )
                        .setArity(            -1          )
                        .setIsOperator(       true        )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );

        Algorithm.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            Operation type = call.getOperation();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if ( d < 0 ) {
                    Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];

                    reduction = Utility.offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];
                } else {
                    Tsr a;
                    if ( d > 1 ) {
                        Tsr[] reduction = Utility.subset(tsrs, 1, 1, d+1);
                        reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ]);
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationContext.get().instance("/") )
                        );
                        a = reduction[ 0 ];
                    } else if ( d == 1 ) a = tsrs[ 1 ];
                    else a = Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);
                    Tsr b;
                    if ( tsrs.length -  d - 2  > 1 ) {
                        Tsr[] reduction = Utility.subset(tsrs, 2, d+2, tsrs.length-(d+2));
                        reduction[ 1 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);
                        reduction[ 0 ] = reduction[ 1 ];
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationContext.get().instance("/") )
                        );
                        b = reduction[ 0 ];
                    } else b = Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);

                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>( device, new Tsr[]{tsrs[ 0 ], a, b}, -1, OperationContext.get().instance("*") )
                    );
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>( device, new Tsr[]{tsrs[ 0 ], tsrs[ 0 ], tsrs[d+1]}, 1, OperationContext.get().instance("/") )
                    );
                    if ( d == 0 ) a.delete();
                    b.delete();
                }
                return alternative;
            } else {
                return alternative;
            }
        };

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<SecondaryNDIConsumer> operationCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) return ( t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] % t2_val[t2Idx.i()];
                    else {
                        return ( t1Idx, t2Idx ) -> {
                            if (d == 0) {
                                return 1 / t2_val[t2Idx.i()];
                            } else {
                                return -(t1_val[ t1Idx.i() ] / Math.pow(t2_val[t2Idx.i()], 2));
                            }
                        };
                    }
                };
        DefaultOperatorCreator<PrimaryNDXConsumer> operationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] % t2_val[ndc2.indexOfIndices( t1Idx )];
                    else {
                        return t1Idx -> {
                            if (d == 0) {
                                return 1 / t2_val[ndc2.indexOfIndices( t1Idx )];
                            } else {
                                return -(t1_val[ndc1.indexOfIndices( t1Idx )] / Math.pow(t2_val[ndc2.indexOfIndices( t1Idx )], 2));
                            }
                        };
                    }
                };

        Operator operator = new Operator()
                   .setADAgentSupplier(
                        ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                )
                .setRJAgent( rja )
                .build();

        setAlgorithm(
                Operator.class,
                operator.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Operator.operate (
                                                                        call.getTensor( 0 ),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        operationXCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Operator.operate (
                                                                        call.getTensor( 0 ),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        operationCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = (call.getTensor( 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTensor( 0 ) != null) ? call.getTensor( 0 ).size() : call.getTensor( 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTensor( offset ) )
                                            .pass( call.getTensor( offset + 1 ) )
                                            .pass( call.getTensor( offset + 2 ) )
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                operator.getKernelSource(), // kernelSource
                                "output = ((int)input1) % ((int)input2);\n",
                                "if ( d==0 ) {\n" +
                                        "    output = 1/input2;\n" +
                                        "} else {\n" +
                                        "    output = -input2 / (float) pow(input1, 2.0f);\n" +
                                        "}",
                                this // OperationType
                        )
                )
        );



        //________________
        // BROADCASTING :

        DefaultOperatorCreator<TertiaryNDIConsumer> creator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] % t2_val[t2Idx.i()];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if (d == 0) {
                                return 1 / t2_val[t2Idx.i()];
                            } else {
                                return
                                        -(t1_val[ t1Idx.i() ]
                                                /
                                                Math.pow(t2_val[t2Idx.i()], 2));
                            }
                        };
                    }
                };

        DefaultOperatorCreator<TertiaryNDXConsumer> creatorX =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    if ( d < 0 ) {
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ ndc1.indexOfIndices( t1Idx ) ] % t2_val[ ndc2.indexOfIndices(t2Idx) ];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if (d == 0) {
                                return 1 / t2_val[ ndc2.indexOfIndices( t2Idx ) ];
                            } else {
                                return - ( t1_val[ ndc1.indexOfIndices( t1Idx ) ] / Math.pow(t2_val[ ndc2.indexOfIndices( t2Idx ) ], 2) );
                            }
                        };
                    }
                };

        Broadcast broadcast = new Broadcast()
            .setBackwardADAnalyzer( call -> true )
            .setForwardADAnalyzer(
                    call -> {
                        Tsr<?> last = null;
                        for ( Tsr<?> t : call.getTensors() ) {
                            if ( last != null && !last.shape().equals(t.shape()) ) return false;
                            last = t; // Note: shapes are cached!
                        }
                        return true;
                    }
            )
            .setADAgentSupplier(
                ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                {
                    Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                    Function mul = Function.Detached.MUL;
                    if ( ctxDerivative != null ) {
                        return new DefaultADAgent( ctxDerivative )
                                .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                .setBackward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                    }
                    Tsr[] inputs = call.getTensors();
                    int d = call.getDerivativeIndex();
                    if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                    else
                    {
                        Tsr deriv = f.derive( inputs, d );
                        return new DefaultADAgent( deriv )
                                .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, deriv}) );
                    }
                }
            )
            .setRJAgent( ( call, goDeeperWith ) -> null )
            .build();

        setAlgorithm(
                Broadcast.class,
                broadcast.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                        ? ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor( 0 ), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        creatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor( 0 ), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        creator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = (call.getTensor( 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTensor( 0 ) != null) ? call.getTensor( 0 ).size() : call.getTensor( 1 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass( call.getTensor( offset ) )
                                            .pass( call.getTensor( offset + 1 ) )
                                            .pass( call.getTensor( offset + 2 ) )
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                broadcast.getKernelSource(), // kernelSource
                                "value = ((int)src1) % ((int)src2);\n",
                                "if (d==0) {\n" +
                                        "    value += (1/handle) * drain;\n" +//TODO: this is probably wrong!
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
                        return t1Idx -> t1_val[ t1Idx.i() ] % value;
                    } else {
                        if (d == 0) return t1Idx -> 1 / value;
                        else return t1Idx -> -value / Math.pow(t1_val[ t1Idx.i() ], 2);
                    }
                };

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[ndc1.indexOfIndices( t1Idx )] % value;
                    } else {
                        if (d == 0) return t1Idx -> 1 / value;
                        else return t1Idx -> - value / Math.pow(t1_val[ndc1.indexOfIndices( t1Idx )], 2);
                    }
                };

        Scalarization scalarization = new Scalarization()
            .setBackwardADAnalyzer( call -> true )
            .setForwardADAnalyzer(
                    call -> {
                        Tsr<?> last = null;
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                    }
            )
            .setADAgentSupplier(
                ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
            )
            .setCallHook( (caller, call ) -> null )
            .setRJAgent( ( call, goDeeperWith ) -> null )
            .build();

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call -> {
                                    double value = call.getTensor( 0 ).value64(2);
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTensor( 0 ).size(),
                                                    (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                    ? ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTensor( 0 ),
                                                                    start, end,
                                                                    scalarXCreator.create(call.getTensors(), value, -1)
                                                            )
                                                    : ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTensor( 0 ),
                                                                    start, end,
                                                                    scalarCreator.create(call.getTensors(), value, -1)
                                                            )
                                            );
                                },
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        new CLImplementation(
                                call -> {
                                    int offset = (call.getTensor( 2 ).isVirtual() || call.getTensor( 2 ).size() == 1)?1:0;
                                    int gwz = call.getTensor( 0 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor( 0 ))
                                            .pass(call.getTensor( 0 ))
                                            .pass((float)call.getTensor(1+offset).value64( 0 ))
                                            .pass( call.getTensor( 0 ).rank() )
                                            .pass( call.getDerivativeIndex() )
                                            .call( gwz );
                                },
                                3,
                                scalarization.getKernelSource(), // kernelSource
                                "output = ((int)input1) % ((int)value);     \n",
                                "if (d==0) {                               \n" +
                                        "    output = 1/value;                           \n" +
                                        "} else {                                        \n" +
                                        "    output = -value /(float)pow(input1, 2.0f);  \n" +
                                        "}",
                                this // OperationType
                        )
                )
        );



        //__________________________
        // RELATED OPERATION TYPES :

        new AbstractOperation(
                new OperationFactory()
                        .setFunction(         ""                 )
                        .setOperator(         ((char) 171) + "%" )
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
                new OperationFactory()
                        .setFunction(         "" )
                        .setOperator(         "%" + ((char) 187)  )
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
    }



    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result %= current;
            }
            return result;
        } else return src[ 0 ].derive( inputs, d );
    }

    @Contract(pure = true)

    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" % ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int d ) {
        return children[ 0 ].getDerivative( d ).toString();
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result %= current;
            }
            return result;
        } else {
            return src[ 0 ].derive( inputs, d, j );// j ?
        }
    }





}
