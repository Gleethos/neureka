package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Operator;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.RecursiveExecutor;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

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

    private final static DefaultOperatorCreator<TertiaryNDAConsumer> _creatorX = (inputs, d )->
    {
        double[] t1_val = inputs[ 1 ].value64();
        double[] t2_val = inputs[ 2 ].value64();
        NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
        NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
        if ( d < 0 ) return ( t0Idx, t1Idx, t2Idx ) ->
                Math.pow(t1_val[ndc1.indexOfIndices( t1Idx )], t2_val[ndc2.indexOfIndices(t2Idx)]);
        else {
            return ( t0Idx, t1Idx, t2Idx ) -> {
                if (d == 0) {
                    double temp = t2_val[ndc2.indexOfIndices(t2Idx)];
                    return temp * Math.pow( t1_val[ndc1.indexOfIndices( t1Idx )], temp - 1 );
                } else {
                    double temp = t1_val[ndc1.indexOfIndices( t1Idx )];
                    return Math.pow( temp, t2_val[ndc2.indexOfIndices(t2Idx)] )  * Math.log(temp);
                }
            };
        }
    };

    public Power()
    {
        super(
                new OperationBuilder()
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

        DefaultOperatorCreator<PrimaryNDAConsumer> operationXCreator = (inputs, d )->
        {
            double[] t1_val = inputs[ 1 ].value64();
            double[] t2_val = inputs[ 2 ].value64();
            NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
            NDConfiguration ndc2 = inputs[ 2 ].getNDConf();
            if ( d < 0 ) return t1Idx ->
                    Math.pow(t1_val[ndc1.indexOfIndices( t1Idx )], t2_val[ndc2.indexOfIndices( t1Idx )]);
            else {
                return t1Idx ->
                {
                    double temp1 = t1_val[ndc1.indexOfIndices( t1Idx )];
                    double temp2 = t2_val[ndc2.indexOfIndices( t1Idx )];
                    if ( d == 0 ) return temp2 * Math.pow( temp1, temp2 - 1 );
                    else return Math.pow( temp1, temp2 ) * Math.log(temp1);
                };
            }
        };

        RecursiveExecutor rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getValOf( Arg.DerivIdx.class );
            Operation type = call.getOperation();

            Tsr alternative = null;
            if ( tsrs.length > 3 )
            {
                if ( d < 0 ) {
                    Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                    alternative = goDeeperWith.execute(
                            call.withTensors( reduction )
                    );
                    tsrs[ 0 ] = reduction[ 0 ];

                    reduction = Utility.offsetted(tsrs, 1);
                    alternative = goDeeperWith.execute(
                                        call.withTensors( reduction )
                            );
                    tsrs[ 0 ] = reduction[ 0 ];
                } else {

                    if ( d==0 ) {
                        Tsr[] reduction = Utility.subset(tsrs, 1,  2, tsrs.length-2);
                        reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ]);
                        alternative = goDeeperWith.execute(
                                            ExecutionCall.of(reduction)
                                                            .andArgs(Arg.DerivIdx.of( -1 ))
                                                            .running(Neureka.get().context().getOperation("*"))
                                                            .on(device)
                                        );
                        Tsr exp = reduction[ 0 ];
                        reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], exp};
                        alternative = goDeeperWith.execute(
                                            ExecutionCall.of(reduction)
                                                            .andArgs(Arg.DerivIdx.of(0))
                                                            .running(type)
                                                            .on( device )
                                        );
                        tsrs[ 0 ] = reduction[ 0 ];
                        exp.delete();
                    } else {
                        Tsr<?>[] reduction = Utility.subset(tsrs, 1,  2, tsrs.length-2);

                        reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ]);
                        alternative = goDeeperWith.execute(
                                                ExecutionCall.of(reduction)
                                                                .andArgs(Arg.DerivIdx.of(d-1))
                                                                .running(Neureka.get().context().getOperation("*"))
                                                                .on(device)
                                        );
                        Tsr<?> inner = reduction[ 0 ];

                        reduction = new Tsr[]{Tsr.Create.newTsrLike(tsrs[ 1 ]), inner, tsrs[d]};
                        alternative = goDeeperWith.execute(
                                                ExecutionCall.of(reduction)
                                                                .andArgs(Arg.DerivIdx.of(-1))
                                                                .running(Neureka.get().context().getOperation("*"))
                                                                .on(device)
                                        );
                        Tsr<?> exp = reduction[ 0 ];

                        reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], exp};
                        alternative = goDeeperWith.execute(
                                ExecutionCall.of(reduction)
                                                .andArgs(Arg.DerivIdx.of(1))
                                                .running(type)
                                                .on(device)
                            );
                        tsrs[ 0 ] = reduction[ 0 ];

                        inner.delete();
                        exp.delete();
                    }
                }
                return alternative;
            } else
                return alternative;



        };

        Operator operator = new Operator( rja )
                                .setSupplyADAgentFor(
                                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                                getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                                )
                                .build();

        setAlgorithm(Operator.class,
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
                                                                        call.getValOf( Arg.DerivIdx.class ),
                                                                        start, end,
                                                                        operationXCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                        : ( start, end ) ->
                                                                Operator.operate (
                                                                        call.getTsrOfType( Number.class, 0 ),
                                                                        call.getTsrOfType( Number.class, 1 ),
                                                                        call.getTsrOfType( Number.class, 2 ),
                                                                        call.getValOf( Arg.DerivIdx.class ),
                                                                        start, end,
                                                                        operationCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                ),
                                3
                        )
                ).setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( operator.getKernelSource() )
                                .activationSource( "output = pow(input1, input2);" )
                                .differentiationSource(
                                        "if (d==0) {                                    \n" +
                                        "    output = input2 * pow(input1, input2-1.0f);  \n" +
                                        "} else {                                         \n" +
                                        "    output = pow(input1, input2) * log(input1);  \n" +
                                        "}"
                                )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call ->
                                        {
                                            int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                            int gwz = (call.getTsrOfType( Number.class, 0 ) != null)
                                                    ? call.getTsrOfType( Number.class, 0 ).size()
                                                    : call.getTsrOfType( Number.class, 1 ).size();
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

        Broadcast broadcast = new Broadcast(rja)
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>)call.getValOf(Arg.Derivative.class);
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
                            Tsr<?> deriv = f.derive( inputs, d );
                            return ADAgent.of( deriv )
                                    .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                    .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, deriv } ) );
                        }
                    }
                )
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
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( broadcast.getKernelSource() )
                                .activationSource( "value += pow(src1, src2);" )
                                .differentiationSource(
                                        "if (d==0) {\n" +
                                        "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                                        "} else {\n" +
                                        "    value += (pow(target, handle) * log(handle)) * drain;\n" +
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
                ( inputs, value, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    if ( d < 0 ) return t1Idx -> Math.pow(t1_val[ t1Idx.i() ], value);
                    else {
                        if (d==0) return t1Idx -> value*Math.pow(t1_val[ t1Idx.i() ], value-1);
                        else return t1Idx -> Math.pow(t1_val[ t1Idx.i() ], value)*Math.log(value);
                    }
                };

        ScalarOperatorCreator<PrimaryNDAConsumer> scalarXCreator =
                ( inputs, value, d ) -> {
                    double[] t1_val = inputs[ 1 ].value64();
                    NDConfiguration ndc1 = inputs[ 1 ].getNDConf();
                    if ( d < 0 ) return t1Idx -> Math.pow(t1_val[ndc1.indexOfIndices( t1Idx )], value);
                    else {
                        if (d==0) return t1Idx -> value*Math.pow(t1_val[ndc1.indexOfIndices( t1Idx )], value-1);
                        else return t1Idx -> Math.pow(t1_val[ndc1.indexOfIndices( t1Idx )], value)*Math.log(value);
                    }
                };

        Scalarization scalarization = new Scalarization()
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
                )
                .setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, rja ) )
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
                                                    (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                    ? ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTsrOfType( Number.class, 0 ),
                                                                    start, end,
                                                                    scalarXCreator.create(call.getTensors(), value, -1)
                                                            )
                                                    : ( start, end ) ->
                                                    Scalarization.scalarize (
                                                            call.getTsrOfType( Number.class, 0 ),
                                                            start, end,
                                                            scalarCreator.create(call.getTensors(), value, -1)
                                                    )
                                            );
                                },
                                3
                        )
                )
                .setImplementationFor(
                        OpenCLDevice.class,
                        CLImplementation.compiler()
                                .arity( 3 )
                                .kernelSource( scalarization.getKernelSource() )
                                .activationSource( "output = pow( input1, value );" )
                                .differentiationSource(
                                        "if ( d == 0 ) {                                      \n" +
                                        "    output = value * pow( input1, value - (float) 1 );   \n" +
                                        "} else {                                             \n" +
                                        "    output = pow( input1, value ) * log( value );        \n" +
                                        "}"
                                )
                                .kernelPostfix( this.getFunction() )
                                .execution(
                                        call -> {
                                            int offset = (call.getTsrOfType( Number.class, 2 ).isVirtual() || call.getTsrOfType( Number.class, 2 ).size() == 1)?1:0;
                                            int gwz = call.getTsrOfType( Number.class, 0 ).size();
                                            call.getDevice().getKernel( call )
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
    public String asDerivative( Function[] children, int derivationIndex) {
        Function a = children[0];
        Function b = Function.of(
                IntStream.range( 1, children.length )
                .mapToObj(i -> children[ i ].toString() )
                .collect(Collectors.joining(" * "))
        );
        boolean aDerivable = a.dependsOn(derivationIndex);
        boolean bDerivable = b.dependsOn(derivationIndex);
        String aAsStr = a.toString();
        String bAsStr = b.toString();
        String first = "";
        if (aDerivable) {
            String aAsDeriv = a.getDerivative(derivationIndex).toString();
            if ( !aAsDeriv.equals("0.0") ) {
                first = ("( "+ bAsStr +" * "+ aAsStr + " ^ (" + bAsStr + " - 1) )");
                if (!aAsDeriv.equals("1.0")) first = aAsDeriv + " * " + first;
            }
        }
        String bAsDeriv = "";
        if (bDerivable) bAsDeriv = b.getDerivative(derivationIndex).toString();
        if ( !bAsDeriv.isEmpty() && !bAsDeriv.equals("1.0") ) bAsDeriv += " * ";
        else bAsDeriv = "";
        String second = "";
        if ( bDerivable ) second = "(ln("+aAsStr+") * "+aAsStr+" ^ "+bAsStr+")";
        String result;
        if ( !first.trim().isEmpty() && !second.trim().isEmpty() ) result = bAsDeriv+"("+first+" + "+second+")";
        else if (!first.trim().isEmpty()) result = bAsDeriv + "("+first+")";
        else if (!second.trim().isEmpty()) result = bAsDeriv + "(" +second + ")";
        else result = bAsDeriv;
        return result;
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
