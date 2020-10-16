package neureka.calculus.backend.operations.operator;

import neureka.Tsr;
import neureka.device.Device;
import neureka.device.host.execution.HostExecutor;
import neureka.device.opencl.execution.CLExecutor;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.implementations.functional.Broadcast;
import neureka.calculus.backend.implementations.functional.Convolution;
import neureka.calculus.backend.implementations.functional.Operator;
import neureka.calculus.backend.implementations.functional.Scalarization;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.implementations.OperationTypeImplementation;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Power extends AbstractOperationType
{

    private final static DefaultOperatorCreator<TertiaryNDXConsumer> _creator = (inputs, d)->
    {
        double[] t1_val = inputs[1].value64();
        double[] t2_val = inputs[2].value64();
        if (d < 0) return (t0Idx, t1Idx, t2Idx) ->
                    Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], t2_val[inputs[2].i_of_idx(t2Idx)]);
        else {
            return (t0Idx, t1Idx, t2Idx) -> {
                if (d == 0) {
                    return t2_val[inputs[2].i_of_idx(t2Idx)]
                            * Math.pow(
                            t1_val[inputs[1].i_of_idx(t1Idx)],
                            t2_val[inputs[2].i_of_idx(t2Idx)] - 1
                    );
                } else {
                    return Math.pow(
                            t1_val[inputs[1].i_of_idx(t1Idx)],
                            t2_val[inputs[2].i_of_idx(t2Idx)]
                    ) * Math.log(t1_val[inputs[1].i_of_idx(t1Idx)]);
                }
            };
        }
    };

    public Power()
    {
        super("power", "^", -1, true, false, true, false);

        setStringifier(
                children ->
                {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get( i ) );
                        if ( i < children.size() - 1 ) reconstructed.append(" ^ ");
                    }
                    return "(" + reconstructed + ")";
                }
        );

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<PrimaryNDXConsumer> operationCreator = (inputs, d)->
        {
            double[] t1_val = inputs[1].value64();
            double[] t2_val = inputs[2].value64();
            if (d < 0) return t1Idx ->
                    Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], t2_val[inputs[2].i_of_idx(t1Idx)]);
            else {
                return t1Idx ->
                {
                    if ( d == 0 ) return
                            t2_val[inputs[2].i_of_idx(t1Idx)] * Math.pow(
                                t1_val[inputs[1].i_of_idx(t1Idx)],
                                t2_val[inputs[2].i_of_idx(t1Idx)] - 1
                            );
                    else return
                            Math.pow(
                                t1_val[inputs[1].i_of_idx(t1Idx)],
                                t2_val[inputs[2].i_of_idx(t1Idx)]
                            ) * Math.log(t1_val[inputs[1].i_of_idx(t1Idx)]);
                };
            }
        };

        OperationTypeImplementation.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            OperationType type = call.getType();

            Tsr alternative = null;
            if ( tsrs.length > 3 )
            {
                if ( d < 0 ) {
                    Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[1], tsrs[2]};
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
                        reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[1]);
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationType.instance("*") )
                        );
                        Tsr exp = reduction[ 0 ];
                        reduction = new Tsr[]{tsrs[ 0 ], tsrs[1], exp};
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, 0, type )
                        );
                        tsrs[ 0 ] = reduction[ 0 ];
                        exp.delete();
                    } else {
                        Tsr[] reduction = Utility.subset(tsrs, 1,  2, tsrs.length-2);

                        reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[1]);
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, d-1, OperationType.instance("*") )
                        );
                        Tsr inner = reduction[ 0 ];

                        reduction = new Tsr[]{Tsr.Create.newTsrLike(tsrs[1]), inner, tsrs[d]};
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationType.instance("*") )
                        );
                        Tsr exp = reduction[ 0 ];

                        reduction = new Tsr[]{tsrs[ 0 ], tsrs[1], exp};
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
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                        defaultImplementation().supplyADAgentFor( f, call, forward )
                )
                .setCallHock( (caller, call) -> null )
                .setRJAgent( rja )
                .setDrainInstantiation(
                    call ->
                    {
                        Tsr[] tsrs = call.getTensors();
                        Device device = call.getDevice();
                        if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                        {
                            int[] shp = tsrs[1].getNDConf().shape();
                            Tsr output = new Tsr( shp, 0.0 );
                            output.setIsVirtual(false);
                            device.add(output);
                            tsrs[ 0 ] = output;
                        }
                        return call;
                    }
                );

        setImplementation(Operator.class,
                operator.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        ( start, end ) ->
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
                                call ->
                                {
                                    int offset = (call.getTensor( 0 ) != null) ? 0 : 1;
                                    int gwz = (call.getTensor( 0 ) != null)
                                            ? call.getTensor( 0 ).size()
                                            : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor( 0 ).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                operator.getKernelSource(), // kernelSource
                                "output = pow(input1, input2);",
                                "if(d==0) {                                    \n" +
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
                                    .withForward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                    .withBackward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr<?> deriv = f.derive(inputs, d);
                            return new DefaultADAgent( deriv )
                                    .withForward( (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                    .withBackward( (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv}) );
                        }
                    }
                )
                .setCallHock( (caller, call) -> null )
                .setRJAgent( rja )
                .setDrainInstantiation(
                        call -> {
                            Tsr[] tsrs = call.getTensors();
                            Device device = call.getDevice();
                            if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                            {
                                int[] shp = tsrs[1].getNDConf().shape();
                                Tsr output = new Tsr( shp, 0.0 );
                                output.setIsVirtual(false);
                                device.add(output);
                                tsrs[ 0 ] = output;
                            }
                            return call;
                        }
                );

        setImplementation(
                Broadcast.class,
                broadcast.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor( 0 ).size(),
                                                        ( start, end ) ->
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
                                    int gwz = (call.getTensor( 0 ) != null) ? call.getTensor( 0 ).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor( 0 ).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                broadcast.getKernelSource(), // kernelSource
                                "value += pow(src1, src2);",
                                "if(d==0){\n" +
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

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarCreator =
                ( inputs, value, d ) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return t1Idx -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value);
                    else {
                        if(d==0) return t1Idx -> value*Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value-1);
                        else return t1Idx -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value)*Math.log(value);
                    }
                };


        Scalarization scalarization = new Scalarization()
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                        defaultImplementation().supplyADAgentFor(f, call, forward)
                )
                .setCallHock( (caller, call) -> null )
                .setRJAgent( rja )
                .setDrainInstantiation(
                    call -> {
                        Tsr[] tsrs = call.getTensors();
                        Device device = call.getDevice();
                        if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                        {
                            int[] shp = tsrs[1].getNDConf().shape();
                            Tsr output = new Tsr( shp, 0.0 );
                            output.setIsVirtual(false);
                            device.add(output);
                            tsrs[ 0 ] = output;
                        }
                        return call;
                    }
                );

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
                                                    ( start, end ) ->
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
                                    int offset = (call.getTensor(2).isVirtual() || call.getTensor(2).size() == 1)?1:0;
                                    int gwz = call.getTensor( 0 ).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor( 0 ))
                                            .pass(call.getTensor( 0 ))
                                            .pass((float)call.getTensor(1+offset).value64( 0 ))
                                            .pass(call.getTensor( 0 ).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
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

        new AbstractOperationType("inv_power_left", ((char) 171) + "^", 3, true, false, false, false) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };
        new AbstractOperationType("inv_power_right", "^" + ((char) 187), 3, true, false, false, false) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };

        // Convolution:

        new AbstractOperationType(
                "power", "p", 2, true, false, false, false
        ){
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ){
                return 0;
            }
        }.setImplementation(
                Convolution.class,
                new Convolution()
                    .setBackwardADAnalyzer( call -> true )
                    .setForwardADAnalyzer(
                            call -> {
                                Tsr last = null;
                                for ( Tsr t : call.getTensors() ) {
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
                                        .withForward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                        .withBackward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                            }
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();
                            if( forward )
                                throw new IllegalArgumentException("Convolution of does not support forward-AD!");
                            else
                            {
                                Tsr<?> localDerivative = f.derive(inputs, d);
                                return new DefaultADAgent( localDerivative )
                                        .withForward( (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, localDerivative}) )
                                        .withBackward( (node, backwardError) -> mul.call(new Tsr[]{backwardError, localDerivative}) );
                            }
                        }
                    )
                    .setCallHock( ( caller, call ) -> null )
                    .setRJAgent( ( call, goDeeperWith ) -> null )
                    .setDrainInstantiation(
                            call -> {
                                Tsr[] tsrs = call.getTensors();
                                int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                                return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                            }
                    )
        )
        .setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get( i ) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" p ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        new AbstractOperationType("", ((char) 171) + "p", 3, true, false, false, false) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };
        new AbstractOperationType("", "p" + ((char) 187), 3, true, false, false, false) {
            @Override
            public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
            }
        };




    }








    // d/dx(f(x)^g(x))=
    // f(x)^g(x) * d/dx(g(x)) * ln(f(x))
    // + f(x)^(g(x)-1) * g(x) * d/dx(f(x))
    @Contract(pure = true)

    @Override
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src.get( 0 ).call(inputs, j);
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get( i ).call(inputs, j);
                result = Math.pow(result, current);
            }
            return result;
        } else {
            double out = 0;
            for ( int si = 0; si < src.size(); si++ ) {
                double b = 1;
                for ( int i = 1; i < src.size(); i++ ) {
                    b *= src.get( i ).call(inputs, j);
                }
                if ( si == 0 ) {
                    out += src.get( 0 ).derive(inputs, d, j) * b * Math.pow(src.get( 0 ).call(inputs, j), b - 1);
                } else {
                    double a = src.get( 0 ).call(inputs, j);
                    out += ( a >= 0 ) ? src.get(si).derive(inputs, d, j) * b * Math.log(a) : 0;
                }
            }
            return out;
        }
    }

    @Contract(pure = true)
    public static double calculate(double[] inputs, int d, List<Function> src) {
        if ( d < 0 ) {
            double result = src.get( 0 ).call(inputs);
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get( i ).call(inputs);
                result = Math.pow(result, current);
            }
            return result;
        } else {
            double b = 1;
            double bd = 0;
            double a = 0;
            for ( int i = 1; i < src.size(); i++ ) {
                double dd = 1;
                a = src.get( i ).call(inputs);
                for ( int di = 1; di < src.size(); di++ ) {
                    if ( di != i ) dd *= a;
                    else dd *= src.get(di).derive(inputs, d);
                }
                bd += dd;
                b *= a;
            }
            double out = 0;
            a = src.get( 0 ).call(inputs);
            out += src.get( 0 ).derive(inputs, d) * b * Math.pow(a, b - 1);
            out += (a >= 0) ? bd *  Math.pow(a, b) * Math.log(a) : 0;
            return out;
        }
    }







}
