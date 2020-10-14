package neureka.calculus.backend.operations.operator;

import neureka.Tsr;
import neureka.device.Device;
import neureka.device.host.execution.HostExecutor;
import neureka.device.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.implementations.functional.Broadcast;
import neureka.calculus.backend.implementations.functional.Operator;
import neureka.calculus.backend.implementations.functional.Scalarization;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.implementations.OperationTypeImplementation;
import org.jetbrains.annotations.Contract;

import java.util.List;


public class Multiplication extends AbstractOperationType {


    private static final DefaultOperatorCreator<TertiaryNDXConsumer> _creator =
            (inputs, d) -> {
                double[] t1_val = inputs[1].value64();
                double[] t2_val = inputs[2].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                } else {
                    return (t0Idx, t1Idx, t2Idx) -> {
                        if (d == 0) return t2_val[inputs[2].i_of_idx(t2Idx)];
                        else return t1_val[inputs[1].i_of_idx(t1Idx)];
                    };
                }
            };

    public Multiplication()
    {
        super(
                "multiply", "*", -1,
                true, false, true, false
        );

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" * ");
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
            OperationType type = call.getType();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if (d < 0) {
                    Tsr[] reduction = new Tsr[]{tsrs[0], tsrs[1], tsrs[2]};
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];

                    reduction = Utility._offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];
                } else {
                    Tsr[] reduction = Utility._without(tsrs, 1+d);
                    if ( reduction.length > 2 ) {
                        reduction[0] = ( reduction[0] == null ) ? Tsr.Create.newTsrLike(tsrs[1]) : reduction[0];
                        alternative = goDeeperWith.apply(
                                new ExecutionCall<>( device, reduction, -1, OperationType.instance("*") )
                        );
                        tsrs[0] = reduction[0];
                    } else tsrs[0] = reduction[1];
                }
                return alternative;
            } else {
                return alternative;
            }
        };

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<PrimaryNDXConsumer> defaultOperatorcreator =
                (inputs, d) -> {
                    inputs[1].setIsVirtual(false);
                    inputs[2].setIsVirtual(false);
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t1Idx)];
                    } else {
                        return t1Idx -> {
                            if ( d == 0 ) return t2_val[inputs[2].i_of_idx(t1Idx)];
                            else return t1_val[inputs[1].i_of_idx(t1Idx)];
                        };
                    }
                };

        Operator operator = new Operator()
                .setBackwardADAnalyzer( call -> true )
        .setForwardADAnalyzer(
                    call -> true
                ).setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if (
                            ctxDerivative != null
                        ) {
                            return new ADAgent(
                                    ctxDerivative
                            ).withForward(
                                    ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative})
                            ).withBackward(
                                   null
                            );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if( forward )
                        {
                            Tsr deriv = f.derive(inputs, d);
                            return new ADAgent(
                                deriv
                            ).withForward(
                                ( t, derivative ) -> mul.call(new Tsr[]{derivative, deriv})
                            ).withBackward(
                                null
                            );
                        }
                        else
                        {
                            Tsr deriv = f.derive(inputs, d);
                            return new ADAgent(
                                deriv
                            ).withForward(
                                (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
                            ).withBackward(
                                (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv})
                            );
                        }
                    }
                ).setCallHock(
                    (caller, call) -> null
                ).setRJAgent(
                    rja
                ).setDrainInstantiation(
                    call -> {
                        Tsr[] tsrs = call.getTensors();
                        Device device = call.getDevice();
                        if ( tsrs[0] == null ) // Creating a new tensor:
                        {
                            int[] shp = tsrs[1].getNDConf().shape();
                            Tsr output = new Tsr( shp, 0.0 );
                            output.setIsVirtual(false);
                            device.add(output);
                            tsrs[0] = output;
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
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Operator.operate (
                                                                        call.getTensor(0),
                                                                        call.getTensor(1),
                                                                        call.getTensor(2),
                                                                        call.getDerivativeIndex(),
                                                                        start, end,
                                                                        defaultOperatorcreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                operator.getKernelSource(), // kernelSource
                                "output = input1 * input2;\n",
                                "if(d==0){output = input2;}else{output = input1;}\n",
                                this // OperationType
                        )
                )
        );


        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast()
                .setBackwardADAnalyzer( call -> true )
        .setForwardADAnalyzer(
                    call -> true
                ).setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if ( ctxDerivative != null ) {
                            return new ADAgent( ctxDerivative )
                                    .withForward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) )
                                    .withBackward( ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative}) );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr deriv = f.derive(inputs, d);
                            return new ADAgent( deriv )
                                    .withForward( (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv}) )
                                    .withBackward( (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv}) );
                        }
                    }
                ).setCallHock(
                    (caller, call) -> null
                ).setRJAgent(
                    rja
                ).setDrainInstantiation(
                    call -> {
                        Tsr[] tsrs = call.getTensors();
                        Device device = call.getDevice();
                        if ( tsrs[0] == null ) // Creating a new tensor:
                        {
                            int[] shp = tsrs[1].getNDConf().shape();
                            Tsr output = new Tsr( shp, 0.0 );
                            output.setIsVirtual(false);
                            device.add(output);
                            tsrs[0] = output;
                        }
                        return call;
                    }
                );

        setImplementation(Broadcast.class,
            broadcast.setExecutor(
                        HostExecutor.class,
                    new HostExecutor(
                            call ->
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTensor(0).size(),
                                                    ( start, end ) ->
                                                            Broadcast.broadcast (
                                                                    call.getTensor(0), call.getTensor(1), call.getTensor(2),
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
                                int offset = (call.getTensor(0) != null) ? 0 : 1;
                                int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                call.getDevice().getKernel(call)
                                        .pass(call.getTensor(offset))
                                        .pass(call.getTensor(offset + 1))
                                        .pass(call.getTensor(offset + 2))
                                        .pass(call.getTensor(0).rank())
                                        .pass(call.getDerivativeIndex())
                                        .call(gwz);
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

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarOperatorCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if ( d < 0 ) return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] * value;
                    else {
                        if ( d == 0 ) return t1Idx -> value;
                        else return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)];
                    }
                };

        Scalarization scalarization = new Scalarization()
                .setBackwardADAnalyzer( call -> true )
        .setForwardADAnalyzer(
                    call -> true
                ).setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                            {
                                Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if (
                            ctxDerivative != null
                        ) {
                            return new ADAgent(
                                    ctxDerivative
                                ).withForward(
                                    ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative})
                                ).withBackward(
                                    null
                                );
                        }
                        Tsr[] inputs = call.getTensors();
                        int d = call.getDerivativeIndex();
                        if( forward )
                        {
                            Tsr deriv = f.derive(inputs, d);
                            return new ADAgent(
                                    deriv
                                ).withForward(
                                    ( t, derivative ) -> mul.call(new Tsr[]{derivative, deriv})
                                ).withBackward(
                                    null
                                );
                        }
                        else
                        {
                            Tsr deriv = f.derive(inputs, d);
                            return new ADAgent(
                                        deriv
                                ).withForward(
                                        (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
                                ).withBackward(
                                        (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv})
                                );
                        }
                    }
                ).setCallHock(
                        (caller, call) -> null
                ).setRJAgent(
                    rja
                ).setDrainInstantiation(
                        call -> {
                            Tsr[] tsrs = call.getTensors();
                            Device device = call.getDevice();
                            if ( tsrs[0] == null ) // Creating a new tensor:
                            {
                                int[] shp = tsrs[1].getNDConf().shape();
                                Tsr output = new Tsr( shp, 0.0 );
                                output.setIsVirtual(false);
                                device.add(output);
                                tsrs[0] = output;
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
                                    double value = call.getTensor(0).value64(2);
                                    call.getDevice().getExecutor()
                                            .threaded (
                                                    call.getTensor(0).size(),
                                                    ( start, end ) ->
                                                            Scalarization.scalarize (
                                                                    call.getTensor(0),
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
                                    int offset = (call.getTensor(2).isVirtual() || call.getTensor(2).size() == 1)?1:0;
                                    int gwz = call.getTensor(0).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(0))
                                            .pass(call.getTensor(0))
                                            .pass((float)call.getTensor(1+offset).value64(0))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                scalarization.getKernelSource(), // kernelSource
                                "output = input1 * value;\n",
                                "if(d==0){output = value;}else{output = input1;}\n",
                                this // OperationType
                        )
                )
        );




        //__________________________
        // RELATED OPERATION TYPES :


        DefaultOperatorCreator<TertiaryNDXConsumer> xCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                };

        Broadcast xBroadcast = new Broadcast()
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
                Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                Function mul = Function.Detached.MUL;
                if (
                    ctxDerivative != null
                ) {
                    return new ADAgent(
                                ctxDerivative
                       ).withForward(
                                ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative})
                       ).withBackward(
                               null
                       );
                }
                Tsr[] inputs = call.getTensors();
                int d = call.getDerivativeIndex();
                if( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                else
                {
                    Tsr deriv = f.derive(inputs, d);
                    return new ADAgent(
                            deriv
                        ).withForward(
                            (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
                        ).withBackward(
                            (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv})
                        );
                }
            }
        ).setCallHock(
                ( caller, call ) -> null
        ).setRJAgent(
                ( call, goDeeperWith ) -> null
        ).setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[0] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                }
        );

        new AbstractOperationType(
                "", ((char) 171) + "*", 3, true, false, false, false
){
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src){
                return 0;
            }
        }.setImplementation(
                Broadcast.class,
                xBroadcast.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor(0), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        xCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
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
                    Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                    Function mul = Function.Detached.MUL;
                    if (
                        ctxDerivative != null
                    ) {
                        return new ADAgent(
                                    ctxDerivative
                           ).withForward(
                                    ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, ctxDerivative})
                           ).withBackward(
                                   null
                           );
                    }
                    Tsr[] inputs = call.getTensors();
                    int d = call.getDerivativeIndex();
                    if( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                    else
                    {
                        Tsr deriv = f.derive(inputs, d);
                        return new ADAgent(
                                deriv
                            ).withForward(
                                (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
                            ).withBackward(
                                (node, backwardError) -> mul.call(new Tsr[]{backwardError, deriv})
                            );
                    }
                }
            ).setCallHock(
                    ( caller, call ) -> null
            ).setRJAgent(
                    ( call, goDeeperWith ) -> null
            ).setDrainInstantiation(
                    call -> {
                        Tsr[] tsrs = call.getTensors();
                        int offset = ( tsrs[0] == null ) ? 1 : 0;
                        return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                    }
            );

        new AbstractOperationType(
                "", "*" + ((char) 187), 3, true, false, false, false
){
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src){
                return 0;
            }
        }.setImplementation(
                Broadcast.class,
                xBroadcast.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTensor(0).size(),
                                                        ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTensor(0), call.getTensor(1), call.getTensor(2),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        xCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
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
    public double calculate(double[] inputs, int j, int d, List<Function> src) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src.get(0).call(inputs, j);
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get(i).call(inputs, j);
                result *= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src.get(0).call(inputs, j);
            ud = src.get(0).derive(inputs, d, j);

            for ( int ji = 1; ji < src.size(); ji++ ) {
                v = src.get(ji).call(inputs, j);
                vd = src.get(ji).derive(inputs, d, j);
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }

    @Contract(pure = true)
    public static double calculate(double[] inputs, int d, List<Function> src) {
        if ( d < 0 ) {
            double result = src.get(0).call(inputs);
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get(i).call(inputs);
                result *= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src.get(0).call(inputs);
            ud = src.get(0).derive(inputs, d);
            for ( int j = 1; j < src.size(); j++ ) {
                v = src.get(j).call(inputs);
                vd = src.get(j).derive(inputs, d);

                ud = u * vd + v * ud;
                u *= v; // ...this step can be avoided (TODO optimize)
            }
            return ud;
        }
    }




}
