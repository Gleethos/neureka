package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.Algorithm;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.Operation;
import neureka.backend.api.operations.OperationContext;
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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


public class Multiplication extends AbstractOperation
{

    private static final DefaultOperatorCreator<TertiaryNDIConsumer> _creator =
            ( inputs, d ) -> {
                double[] t1_val = inputs[ 1 ].value64();
                double[] t2_val = inputs[ 2 ].value64();
                if ( d < 0 ) {
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] * t2_val[t2Idx.i()];
                } else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (d == 0) return t2_val[t2Idx.i()];
                        else return t1_val[ t1Idx.i() ];
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
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.i_of_idx( t1Idx )] * t2_val[ndc2.i_of_idx(t2Idx)];
                } else {
                    return ( t0Idx, t1Idx, t2Idx ) -> {
                        if (d == 0) return t2_val[ndc2.i_of_idx(t2Idx)];
                        else return t1_val[ndc1.i_of_idx( t1Idx )];
                    };
                }
            };

    public Multiplication()
    {
        super(
                "multiply", "*", -1,
                true, false, true, false
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
                    Tsr[] reduction = Utility.without(tsrs, 1+d);
                    if ( reduction.length > 2 ) {
                        reduction[ 0 ] = ( reduction[ 0 ] == null ) ? Tsr.Create.newTsrLike(tsrs[ 1 ]) : reduction[ 0 ];
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationContext.get().instance("*") )
                        );
                        tsrs[ 0 ] = reduction[ 0 ];
                    } else tsrs[ 0 ] = reduction[ 1 ];
                }
                return alternative;
            } else {
                return alternative;
            }
        };

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


        DefaultOperatorCreator<PrimaryNDXConsumer> defaultOperatorXcreator =
                ( inputs, d ) -> {
                    inputs[ 1 ].setIsVirtual( false );
                    inputs[ 2 ].setIsVirtual( false );
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )] * t2_val[ndc2.i_of_idx( t1Idx )];
                    } else {
                        return t1Idx -> {
                            if ( d == 0 ) return t2_val[ndc2.i_of_idx( t1Idx )];
                            else return t1_val[ndc1.i_of_idx( t1Idx )];
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
                                                                        defaultOperatorXcreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Operator.operate (
                                                                        call.getTensor( 0 ),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        defaultOperatorcreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                "output = input1 * input2;\n",
                                "if (d==0) {output = input2;}else{output = input1;}\n",
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
                        Tsr ctxDerivative = (Tsr)call.getAt( "derivative" );
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
                .setRJAgent( rja )
                .build();

        setAlgorithm(Broadcast.class,
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
                            "value = src1 * src2;\n",
                            "value += handle * drain;\n",
                            this // OperationType
                    )
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

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarOperatorXCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )] * value;
                    else {
                        if ( d == 0 ) return t1Idx -> value;
                        else return t1Idx -> t1_val[ndc1.i_of_idx( t1Idx )];
                    }
                };

        Scalarization scalarization = new Scalarization()
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if ( ctxDerivative != null ) {
                            return new DefaultADAgent( ctxDerivative )
                                    .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                    .setBackward( null );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if ( forward )
                        {
                            Tsr deriv = f.derive( inputs, d );
                            return new DefaultADAgent(  deriv )
                                    .setForward( (t, derivative ) -> mul.call(new Tsr[]{derivative, deriv}) )
                                    .setBackward( null );
                        }
                        else
                        {
                            Tsr deriv = f.derive( inputs, d );
                            return new DefaultADAgent(  deriv )
                                    .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                    .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, deriv}) );
                        }
                    }
                )
                .setCallHook( (caller, call ) -> null )
                .setRJAgent( rja )
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
                                "output = input1 * value;\n",
                                "if (d==0) {output = value;}else{output = input1;}\n",
                                this // OperationType
                        )
                )
        );




        //__________________________
        // RELATED OPERATION TYPES :


        DefaultOperatorCreator<TertiaryNDIConsumer> xBCCreator =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ t1Idx.i() ] * t2_val[t2Idx.i()];
                };

        DefaultOperatorCreator<TertiaryNDXConsumer> xBCCreatorX =
                ( inputs, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    double[] t2_val = inputs[ 2 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
                    return ( t0Idx, t1Idx, t2Idx ) -> t1_val[ndc1.i_of_idx( t1Idx )] * t2_val[ndc2.i_of_idx(t2Idx)];
                };

        Broadcast xBroadcast = new Broadcast()
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
                Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                Function mul = Function.Detached.MUL;
                if ( ctxDerivative != null ) {
                    return new DefaultADAgent( ctxDerivative )
                            .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                            .setBackward( null );
                }
                Tsr[] inputs = call.getTensors();
                int d = call.getDerivativeIndex();
                if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                else
                {
                    Tsr deriv = f.derive( inputs, d );
                    return new DefaultADAgent(
                            deriv
                        ).setForward(
                            ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv})
                        ).setBackward(
                            ( node, backwardError ) -> mul.call(new Tsr[]{backwardError, deriv})
                        );
                }
            }
        )
        .setCallHook( (caller, call ) -> null )
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationContext.get().instance("idy") );
                }
        )
        .build();

        new AbstractOperation(
                "", ((char) 171) + "*", 3, true, false, false, false
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
                return 0;
            }
        }.setAlgorithm(
                Broadcast.class,
                xBroadcast.setImplementationFor(
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
                                                                        xBCCreatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor( 0 ), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        xBCCreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                xBroadcast.getKernelSource(), // kernelSource
                                "value = src1 * src2;\n",
                                "value += handle * drain;\n",
                                this // OperationType
                        )
                )
        );

        xBroadcast = new Broadcast()
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
                    Tsr<?> ctxDerivative = (Tsr<?>)call.getAt("derivative");
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
            .setCallHook( (caller, call ) -> null )
            .setRJAgent( ( call, goDeeperWith ) -> null )
            .setDrainInstantiation(
                    call -> {
                        Tsr[] tsrs = call.getTensors();
                        int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                        return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationContext.get().instance("idy") );
                    }
            )
            .build();

        new AbstractOperation(
                "", "*" + ((char) 187), 3, true, false, false, false
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
                return 0;
            }
        }.setAlgorithm(
                Broadcast.class,
                xBroadcast.setImplementationFor(
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
                                                                        xBCCreatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                        : ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor( 0 ), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        xBCCreator.create(call.getTensors(), call.getDerivativeIndex())
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
                                xBroadcast.getKernelSource(), // kernelSource
                                "value = src1 * src2;\n",
                                "value += handle * drain;\n",
                                this // OperationType
                        )
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
    public String asDerivative( Function[] children, int d ) {
        return Arrays.stream( children )
                .filter( child -> child.dependsOn( d ) )
                .map( child -> {
                            String derivative = child.getDerivative( d ).toString();
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
