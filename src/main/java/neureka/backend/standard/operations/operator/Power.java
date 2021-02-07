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

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Power extends AbstractOperation
{

    private final static DefaultOperatorCreator<TertiaryNDIConsumer> _creator = ( inputs, d )->
    {
        double[] t1_val = inputs[ 1 ].value64();
        double[] t2_val = inputs[ 2 ].value64();
        if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) -> Math.pow(t1_val[ t1Idx.i() ], t2_val[t2Idx.i()]);
        else {
            return ( t0Idx, t1Idx, t2Idx ) -> {
                if (d == 0) {
                    return t2_val[t2Idx.i()]
                            * Math.pow(
                            t1_val[ t1Idx.i() ],
                            t2_val[t2Idx.i()] - 1
                    );
                } else {
                    return Math.pow(
                            t1_val[ t1Idx.i() ],
                            t2_val[t2Idx.i()]
                    ) * Math.log(t1_val[ t1Idx.i() ]);
                }
            };
        }
    };

    private final static DefaultOperatorCreator<TertiaryNDXConsumer> _creatorX = ( inputs, d )->
    {
        double[] t1_val = inputs[ 1 ].value64();
        double[] t2_val = inputs[ 2 ].value64();
        NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
        NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
        if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) ->
                Math.pow(t1_val[ndc1.i_of_idx( t1Idx )], t2_val[ndc2.i_of_idx(t2Idx)]);
        else {
            return ( t0Idx, t1Idx, t2Idx ) -> {
                if (d == 0) {
                    double temp = t2_val[ndc2.i_of_idx(t2Idx)];
                    return temp * Math.pow( t1_val[ndc1.i_of_idx( t1Idx )], temp - 1 );
                } else {
                    double temp = t1_val[ndc1.i_of_idx( t1Idx )];
                    return Math.pow( temp, t2_val[ndc2.i_of_idx(t2Idx)] )  * Math.log(temp);
                }
            };
        }
    };

    public Power()
    {
        super(
                new OperationFactory()
                        .setFunction(         "power"    )
                        .setOperator(         "^"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<SecondaryNDIConsumer> operationCreator = ( inputs, d )->
        {
            double[] t1_val = inputs[ 1 ].value64();
            double[] t2_val = inputs[ 2 ].value64();
            if ( d < 0 ) return ( t1Idx, t2Idx ) ->
                    Math.pow(t1_val[ t1Idx.i() ], t2_val[t2Idx.i()]);
            else {
                return ( t1Idx, t2Idx ) ->
                {
                    if ( d == 0 ) return
                            t2_val[t2Idx.i()] * Math.pow(
                                t1_val[ t1Idx.i() ],
                                t2_val[t2Idx.i()] - 1
                            );
                    else return
                            Math.pow(
                                t1_val[ t1Idx.i() ],
                                t2_val[t2Idx.i()]
                            ) * Math.log(t1_val[ t1Idx.i() ]);
                };
            }
        };

        DefaultOperatorCreator<PrimaryNDXConsumer> operationXCreator = ( inputs, d )->
        {
            double[] t1_val = inputs[ 1 ].value64();
            double[] t2_val = inputs[ 2 ].value64();
            NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
            NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
            if ( d < 0 ) return t1Idx ->
                    Math.pow(t1_val[ndc1.i_of_idx( t1Idx )], t2_val[ndc2.i_of_idx( t1Idx )]);
            else {
                return t1Idx ->
                {
                    double temp1 = t1_val[ndc1.i_of_idx( t1Idx )];
                    double temp2 = t2_val[ndc2.i_of_idx( t1Idx )];
                    if ( d == 0 ) return temp2 * Math.pow( temp1, temp2 - 1 );
                    else return Math.pow( temp1, temp2 ) * Math.log(temp1);
                };
            }
        };

        Algorithm.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            Operation type = call.getOperation();

            Tsr alternative = null;
            if ( tsrs.length > 3 )
            {
                if ( d < 0 ) {
                    Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                    alternative = goDeeperWith.apply(
                            call.withNew( reduction )
                    );
                    tsrs[ 0 ] = reduction[ 0 ];

                    reduction = Utility.offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            call.withNew( reduction )
                            );
                    tsrs[ 0 ] = reduction[ 0 ];
                } else {

                    if ( d==0 ) {
                        Tsr[] reduction = Utility.subset(tsrs, 1,  2, tsrs.length-2);
                        reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ]);
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationContext.get().instance("*") )
                        );
                        Tsr exp = reduction[ 0 ];
                        reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], exp};
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, 0, type )
                        );
                        tsrs[ 0 ] = reduction[ 0 ];
                        exp.delete();
                    } else {
                        Tsr[] reduction = Utility.subset(tsrs, 1,  2, tsrs.length-2);

                        reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ]);
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, d-1, OperationContext.get().instance("*") )
                        );
                        Tsr inner = reduction[ 0 ];

                        reduction = new Tsr[]{Tsr.Create.newTsrLike(tsrs[ 1 ]), inner, tsrs[d]};
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationContext.get().instance("*") )
                        );
                        Tsr exp = reduction[ 0 ];

                        reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], exp};
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, 1, type )
                        );
                        tsrs[ 0 ] = reduction[ 0 ];

                        inner.delete();
                        exp.delete();
                    }
                }
                return alternative;
            } else {
                return alternative;
            }


        };

        Operator operator = new Operator()
                   .setADAgentSupplier(
                        ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                )
                .setRJAgent( rja )
                .build();

        setAlgorithm(Operator.class,
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
                                call ->
                                {
                                    int offset = (call.getTensor( 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTensor( 0 ) != null)
                                            ? call.getTensor( 0 ).size()
                                            : call.getTensor( 1 ).size();
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
                                "output = pow(input1, input2);",
                                "if (d==0) {                                    \n" +
                                        "    output = input2 * pow(input1, input2-1.0f);  \n" +
                                        "} else {                                         \n" +
                                        "    output = pow(input1, input2) * log(input1);  \n" +
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
                                    .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                    .setBackward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr<?> deriv = f.derive( inputs, d );
                            return new DefaultADAgent( deriv )
                                    .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                    .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, deriv}) );
                        }
                    }
                )
                .setRJAgent( rja )
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
                                "value += pow(src1, src2);",
                                "if (d==0) {\n" +
                                        "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                                        "} else {\n" +
                                        "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                                        "}",
                                this // OperationType
                        )
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        ScalarOperatorCreator<PrimaryNDIConsumer> scalarCreator =
                ( inputs, value, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return t1Idx -> Math.pow(t1_val[ t1Idx.i() ], value);
                    else {
                        if (d==0) return t1Idx -> value*Math.pow(t1_val[ t1Idx.i() ], value-1);
                        else return t1Idx -> Math.pow(t1_val[ t1Idx.i() ], value)*Math.log(value);
                    }
                };

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarXCreator =
                ( inputs, value, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> Math.pow(t1_val[ndc1.i_of_idx( t1Idx )], value);
                    else {
                        if (d==0) return t1Idx -> value*Math.pow(t1_val[ndc1.i_of_idx( t1Idx )], value-1);
                        else return t1Idx -> Math.pow(t1_val[ndc1.i_of_idx( t1Idx )], value)*Math.log(value);
                    }
                };

        Scalarization scalarization = new Scalarization()
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                        getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
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
                                "output = pow(input1, value);",
                                "if ( d==0 ) {                                     \n" +
                                        "    output = value * pow(input1, value-(float)1 );   \n" +
                                        "} else {                                             \n" +
                                        "    output = pow(input1, value) * log(value);        \n" +
                                        "}",
                                this // OperationType
                        )
                )
        );




        //__________________________
        // RELATED OPERATION TYPES :

        new AbstractOperation(
                new OperationFactory()
                        .setFunction(         "inv_power_left"   )
                        .setOperator(         ((char) 171) + "^" )
                        .setArity(            3                  )
                        .setIsOperator(       true               )
                        .setIsIndexer(        false              )
                        .setIsDifferentiable( false              )
                        .setIsInline(         false              )
        ) {
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
        new AbstractOperation("inv_power_right", "^" + ((char) 187), 3, true, false, false, false) {
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
                "power", "p", 2, true, false, false, false
                ) {
            @Override
            public String stringify(String[] children) {
                StringBuilder reconstructed = new StringBuilder();
                for ( int i = 0; i < children.length; ++i ) {
                    reconstructed.append( children[ i ] );
                    if ( i < children.length - 1 ) {
                        reconstructed.append(" p ");
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
                            Tsr<?> ctxDerivative = (Tsr<?>) call.getAt("derivative");
                            Function mul = Function.Detached.MUL;
                            if ( ctxDerivative != null ) {
                                return new DefaultADAgent( ctxDerivative )
                                        .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                        .setBackward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
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
                    .setCallHook( (caller, call ) -> null )
                    .setRJAgent( ( call, goDeeperWith ) -> null )
                    .setDrainInstantiation(
                            call -> {
                                Tsr[] tsrs = call.getTensors();
                                int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                                return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationContext.get().instance("idy") );
                            }
                    )
                    .build()
        );

        new AbstractOperation("", ((char) 171) + "p", 3, true, false, false, false) {
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
        new AbstractOperation("", "p" + ((char) 187), 3, true, false, false, false) {
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








    // d/dx(f(x)^g(x))=
    // f(x)^g(x) * d/dx(g(x)) * ln(f(x))
    // + f(x)^(g(x)-1) * g(x) * d/dx(f(x))
    @Contract(pure = true)

    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) reconstructed.append(" ^ ");
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int d ) {
        Function a = children[0];
        Function b = Function.create(
                IntStream.range( 1, children.length )
                .mapToObj(i -> children[ i ].toString() )
                .collect(Collectors.joining(" * "))
        );
        String aAsStr = a.toString();
        String bAsStr = b.toString();
        String first = bAsStr + aAsStr + " ^ (" + bAsStr + " - 1)";
        String second = "ln("+aAsStr+") * "+aAsStr+" ^ "+bAsStr;
        return "( "+first+" )+("+second+")";
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result = Math.pow(result, current);
            }
            return result;
        } else {
            double out = 0;
            for ( int si = 0; si < src.length; si++ ) {
                double b = 1;
                for ( int i = 1; i < src.length; i++ ) {
                    b *= src[ i ].call( inputs, j );
                }
                if ( si == 0 ) {
                    out += src[ 0 ].derive( inputs, d, j ) * b * Math.pow(src[ 0 ].call( inputs, j ), b - 1);
                } else {
                    double a = src[ 0 ].call( inputs, j );
                    out += ( a >= 0 ) ? src[ si ].derive( inputs, d, j ) * b * Math.log(a) : 0;
                }
            }
            return out;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result = Math.pow(result, current);
            }
            return result;
        } else {
            double b = 1;
            double bd = 0;
            double a = 0;
            for ( int i = 1; i < src.length; i++ ) {
                double dd = 1;
                a = src[ i ].call( inputs );
                for ( int di = 1; di < src.length; di++ ) {
                    if ( di != i ) dd *= a;
                    else dd *= src[ di ].derive( inputs, d );
                }
                bd += dd;
                b *= a;
            }
            double out = 0;
            a = src[ 0 ].call( inputs );
            out += src[ 0 ].derive( inputs, d ) * b * Math.pow(a, b - 1);
            out += (a >= 0) ? bd *  Math.pow(a, b) * Math.log(a) : 0;
            return out;
        }
    }







}
