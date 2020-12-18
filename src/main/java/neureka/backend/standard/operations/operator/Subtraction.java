package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.implementations.OperationTypeImplementation;
import neureka.backend.standard.implementations.Broadcast;
import neureka.backend.standard.implementations.Operator;
import neureka.backend.standard.implementations.Scalarization;
import neureka.backend.api.operations.AbstractOperationType;
import neureka.backend.api.operations.OperationType;
import neureka.devices.Device;
import neureka.devices.host.execution.HostExecutor;
import neureka.devices.opencl.execution.CLExecutor;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Subtraction extends AbstractOperationType
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

    private static final DefaultOperatorCreator<TertiaryNDXConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                if ( d < 0 ) {
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.i_of_idx( t1Idx )] - t2_val[ndc2.i_of_idx(t2Idx)];
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
                "subtract", "-", -1, true, false, true, false
        );

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get( i ) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" - ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        OperationTypeImplementation.RecursiveJunctionAgent rja =
        (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            OperationType type = call.getOperation();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if ( d < 0 ) {
                    Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<Device>(device, reduction, d, type)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];

                    reduction = Utility.offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<Device>(device, reduction, d, type)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];
                } else {
                    tsrs[ 0 ] = Tsr.Create.newTsrLike(tsrs[ 1 ]).setValue((d==0)?1.0f:-1.0f);
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
                    if ( d < 0 ) {
                        return ( t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] - t2_val[t2Idx.i()];
                    } else return ( t1Idx, t2Idx ) -> ( d == 0 ) ? 1.0 : -1.0;
                };
        DefaultOperatorCreator<PrimaryNDXConsumer> operationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )] - t2_val[ndc2.i_of_idx( t1Idx )];
                    } else return t1Idx -> ( d == 0 ) ? 1.0 : -1.0;
                };

        Operator operator = new Operator()
                   .setADAgentSupplier(
                        ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                                getDefaultImplementation().supplyADAgentFor( f, call, forward )
                )
                .setRJAgent( rja )
                .build();

        setImplementation(
                Operator.class,
                operator.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
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
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
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
                                "output = input1 - input2;  \n",
                                "if(d==0) {                 \n" +//drn and src2 switch:
                                        "    output = 1;              \n" +
                                        "} else {                     \n" +
                                        "    output = -1;               " +
                                        "}",
                                this // OperationType
                        )
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

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarOperatorXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )] - value;
                    else if ( d == 0 ) return t1Idx -> 1; else return t1Idx -> -1;
                };

        Scalarization scalarization = new Scalarization()
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    getDefaultImplementation().supplyADAgentFor( f, call, forward )
                )
                .setCallHook( (caller, call ) -> null )
                .setRJAgent( rja )
                .build();

        setImplementation(
                Scalarization.class,
                scalarization.setExecutor (
                        HostExecutor.class,
                        new HostExecutor (
                                call -> {
                                    int offset = (call.getTensor( 2 ).isVirtual() || call.getTensor( 2 ).size() == 1) ? 1 : 0;
                                    double value = call.getTensor(1+offset).value64( 0 );
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTensor( 0 ).size(),
                                                    (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                    ? ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTensor( 0 ),
                                                                    start, end,
                                                                    scalarOperatorXCreator.create(call.getTensors(), value, -1)
                                                            )
                                                    : ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTensor( 0 ),
                                                                    start, end,
                                                                    scalarOperatorCreator.create(call.getTensors(), value, -1)
                                                            )
                                            );
                                },
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
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
                                "output = input1 - value;\n",
                                "if(d==0) {     \n" +//drn and src2 switch:
                                        "    output = 1;  \n" +
                                        "} else {         \n" +
                                        "    output = -1;   " +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast()
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                        ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                        {
                            Tsr<?> ctxDerivative = (Tsr<?>)call.getAt("derivative");
                            Function mul = Function.Detached.MUL;
                            if ( ctxDerivative != null ) {
                                return new DefaultADAgent( ctxDerivative )
                                        .withForward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                        .withBackward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                            }
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();
                            if( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                            else
                            {
                                Tsr deriv = f.derive( inputs, d );
                                return new DefaultADAgent( deriv )
                                        .withForward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                        .withBackward( ( node, backwardError ) -> mul.call(new Tsr[]{backwardError, deriv}) );
                            }
                        }
                )
                .setRJAgent( rja )
                .build();

        setImplementation (
                Broadcast.class,
                        broadcast
                        .setExecutor(
                            HostExecutor.class,
                            new HostExecutor(
                                    call ->
                                            call.getDevice().getExecutor()
                                                    .threaded (
                                                            call.getTensor( 0 ).size(),
                                                            (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                                    ? ( start, end ) ->
                                                                    Broadcast.broadcast (
                                                                            call.getTensor( 0 ), call.getTensor(1), call.getTensor(2),
                                                                            call.getDerivativeIndex(), start, end,
                                                                            _creatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                    )
                                                                    : ( start, end ) ->
                                                                    Broadcast.broadcast (
                                                                            call.getTensor( 0 ), call.getTensor(1), call.getTensor(2),
                                                                            call.getDerivativeIndex(), start, end,
                                                                            _creator.create(call.getTensors(), call.getDerivativeIndex())
                                                                    )
                                                    ),
                                    3
                            )
                    ).setExecutor(
                            CLExecutor.class,
                            new CLExecutor(
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
                                    "value = src1 - src2;\n",
                                    "value += handle - drain;\n",
                                    this // OperationType
                            )
                    )
                );

        //______________________
        // RELATED OPERATIONS :

        new AbstractOperationType(
                "", ((char) 171) + "-", 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };
        new AbstractOperationType(
                "", "-" + ((char) 187), 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };

        // Convolution:


        new AbstractOperationType(
                "", "s", 2, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        }.setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get( i ) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" s ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        new AbstractOperationType(
                "", ((char) 171) + "s", 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };
        new AbstractOperationType(
                "", "s" + ((char) 187), 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };


    }


    @Contract(pure = true)

    @Override
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src.get( 0 ).call( inputs, j );
            for ( int Vi = 1; Vi < src.size(); Vi++ ) {
                final double current = src.get(Vi).call( inputs, j );
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.size(); ++i ) {
                if (i == 0) {
                    derivative += src.get( i ).derive( inputs, d, j );
                } else {
                    derivative -= src.get( i ).derive( inputs, d, j );
                }
            }
            return derivative;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, List<Function> src ) {
        if ( d < 0 ) {
            double result = src.get( 0 ).call( inputs );
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get( i ).call( inputs );
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.size(); ++i ) {
                if ( i == 0 ) {
                    derivative += src.get( i ).derive( inputs, d );
                } else {
                    derivative -= src.get( i ).derive( inputs, d );
                }
            }
            return derivative;
        }
    }



}
