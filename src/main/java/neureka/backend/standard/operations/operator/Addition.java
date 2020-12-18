package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.devices.Device;
import neureka.devices.host.execution.HostExecutor;
import neureka.devices.opencl.execution.CLExecutor;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.backend.standard.implementations.Broadcast;
import neureka.backend.standard.implementations.Convolution;
import neureka.backend.standard.implementations.Operator;
import neureka.backend.standard.implementations.Scalarization;
import neureka.backend.api.operations.AbstractOperationType;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.OperationType;
import neureka.backend.api.implementations.OperationTypeImplementation;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Addition extends AbstractOperationType {

    private static final DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] + t2_val[t2Idx.i()];
                else return ( t0Idx, t1Idx, t2Idx ) -> 1.0;
            };

    private static final DefaultOperatorCreator<TertiaryNDXConsumer> _creatorX =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.i_of_idx( t1Idx )] + t2_val[ndc2.i_of_idx(t2Idx)];
                else return ( t0Idx, t1Idx, t2Idx ) -> 1.0;
            };


    private static final Broadcast _broadcast = new Broadcast()
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
        ).setADAgentSupplier(
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
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .build();

    public Addition()
    {
        super (
                "add",
                "+",
                -1,
                true,
                false,
                true,
                false
        );

        setStringifier(
            children -> {
                StringBuilder reconstructed = new StringBuilder();
                for ( int i = 0; i < children.size(); ++i ) {
                    reconstructed.append( children.get( i ) );
                    if ( i < children.size() - 1 ) {
                        reconstructed.append(" + ");
                    }
                }
                return "(" + reconstructed + ")";
            }
        );

        OperationTypeImplementation.RecursiveJunctionAgent rja = (call, goDeeperWith)->
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
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];

                    reduction = Utility.offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[ 0 ] = reduction[ 0 ];
                } else {
                    tsrs[ 0 ] = Tsr.Create.newTsrLike(tsrs[ 1 ]).setValue(1.0f);
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
                    if ( d < 0 ) return ( t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] + t2_val[t2Idx.i()];
                    else return ( t1Idx, t2Idx ) -> 1.0;
                };

        DefaultOperatorCreator<PrimaryNDXConsumer> operationXCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )] + t2_val[ndc2.i_of_idx( t1Idx )];
                    else return t1Idx -> 1.0;
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
                operator
                        .setExecutor(
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
                                "output = input1 + input2;\n",
                                "output = 1;\n",
                                this // OperationType
                        )
                )
        );

        //________________
        // BROADCASTING :

        setImplementation(Broadcast.class,
                _broadcast
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
                                _broadcast.getKernelSource(), // kernelSource
                                "value = src1 + src2;\n",
                                "value += 1 * drain;\n",
                                this // OperationType
                        )
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

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

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return t1Idx -> t1_val[ t1Idx.i() ] + value;
                    else {
                        if (d == 0) return t1Idx -> 1;
                        else return t1Idx -> 1;
                    }
                };

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )] + value;
                    else {
                        if (d == 0) return t1Idx -> 1;
                        else return t1Idx -> 1;
                    }
                };

        setImplementation(
                Scalarization.class,
                scalarization.setExecutor(
                        HostExecutor.class,
                                new HostExecutor(
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
                                "output = input1 + value;\n",
                                "output = 1;\n",
                                this // OperationType
                        )
                )
        );

        //__________________________
        // RELATED OPERATION TYPES :

        new AbstractOperationType(
                "", ((char) 171) + "+", 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
                return 0;
            }
        }.setImplementation(Broadcast.class, _broadcast);

        new AbstractOperationType(
                "", "+" + ((char) 187), 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
                return 0;
            }
        }.setImplementation(Broadcast.class, _broadcast);

        // Convolutoion:

        new AbstractOperationType(
                "add", "a", 2, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
                return 0;
            }
        }
        .setImplementation(
                Convolution.class,
                new Convolution()
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
                            Tsr<?> ctxDerivative = (Tsr<?>) call.getAt("derivative");
                            Function mul = Function.Detached.MUL;
                            if ( ctxDerivative != null ) {
                                return new DefaultADAgent( ctxDerivative )
                                        .withForward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                        .withBackward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                            }
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();
                            if( forward )
                                throw new IllegalArgumentException("Convolution of does not support forward-AD!");
                            else
                            {
                                Tsr<?> localDerivative = f.derive( inputs, d );
                                return new DefaultADAgent( localDerivative )
                                    .withForward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, localDerivative}) )
                                    .withBackward( ( node, backwardError ) -> mul.call(new Tsr[]{backwardError, localDerivative}) );
                            }
                        }
                    )
                    .setCallHook( (caller, call ) -> null )
                    .setRJAgent( ( call, goDeeperWith ) -> null )
                    .setDrainInstantiation(
                            call -> {
                                Tsr[] tsrs = call.getTensors();
                                int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                                return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                            }
                    )
                    .build()
        )
        .setStringifier(
            children -> {
                StringBuilder reconstructed = new StringBuilder();
                for ( int i = 0; i < children.size(); ++i ) {
                    reconstructed.append( children.get( i ) );
                    if ( i < children.size() - 1 ) {
                        reconstructed.append(" a ");
                    }
                }
                return "(" + reconstructed + ")";
            }
        );

        new AbstractOperationType(
                "", ((char) 171) + "a", 3, true, false, false, false
        ) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };
        new AbstractOperationType(
                "", "a" + ((char) 187), 3, true, false, false, false
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
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get( i ).call( inputs, j );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.size(); ++i ) {
                derivative += src.get( i ).derive( inputs, d, j );
            }
            return derivative;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, List<Function> src ) {
        if ( d < 0 ) {
            double result = src.get( 0 ).call( inputs );
            for ( int Vi = 1; Vi < src.size(); Vi++ ) {
                final double current = src.get(Vi).call( inputs );
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
