package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.algorithms.Algorithm;
import neureka.backend.api.operations.AbstractOperation;
import neureka.devices.Device;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Operator;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.Operation;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Modulo extends AbstractOperation {

    public Modulo()
    {

        super(
                "modulo", "%", -1, true, false, true, false
        );

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get( i ) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" % ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
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
                                new ExecutionCall<>( device, reduction, -1, Operation.instance("/") )
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
                                new ExecutionCall<>( device, reduction, -1, Operation.instance("/") )
                        );
                        b = reduction[ 0 ];
                    } else b = Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);

                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>( device, new Tsr[]{tsrs[ 0 ], a, b}, -1, Operation.instance("*") )
                    );
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>( device, new Tsr[]{tsrs[ 0 ], tsrs[ 0 ], tsrs[d+1]}, 1, Operation.instance("/") )
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
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )] % t2_val[ndc2.i_of_idx( t1Idx )];
                    else {
                        return t1Idx -> {
                            if (d == 0) {
                                return 1 / t2_val[ndc2.i_of_idx( t1Idx )];
                            } else {
                                return -(t1_val[ndc1.i_of_idx( t1Idx )] / Math.pow(t2_val[ndc2.i_of_idx( t1Idx )], 2));
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
                        return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ ndc1.i_of_idx( t1Idx ) ] % t2_val[ ndc2.i_of_idx(t2Idx) ];
                    } else {
                        return ( t0Idx, t1Idx, t2Idx ) -> {
                            if (d == 0) {
                                return 1 / t2_val[ ndc2.i_of_idx( t2Idx ) ];
                            } else {
                                return - ( t1_val[ ndc1.i_of_idx( t1Idx ) ] / Math.pow(t2_val[ ndc2.i_of_idx( t2Idx ) ], 2) );
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
                        return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )] % value;
                    } else {
                        if (d == 0) return t1Idx -> 1 / value;
                        else return t1Idx -> - value / Math.pow(t1_val[ndc1.i_of_idx( t1Idx )], 2);
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
                "", ((char) 171) + "%", 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };
        new AbstractOperation(
                "", "%" + ((char) 187), 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };
    }



    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, List<Function> src ) {
        if ( d < 0 ) {
            double result = src.get( 0 ).call( inputs );
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get( i ).call( inputs );
                result %= current;
            }
            return result;
        } else return src.get( 0 ).derive( inputs, d );
    }

    @Contract(pure = true)

    @Override
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src.get( 0 ).call( inputs, j );
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get( i ).call( inputs, j );
                result %= current;
            }
            return result;
        } else {
            return src.get( 0 ).derive( inputs, d, j );// j ?
        }
    }





}
