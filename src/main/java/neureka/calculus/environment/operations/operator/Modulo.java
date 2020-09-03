package neureka.calculus.environment.operations.operator;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.AbstractOperationType;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.*;
import org.jetbrains.annotations.Contract;

import java.util.List;

public class Modulo extends AbstractOperationType {

    public Modulo()
    {

        super(
                "modulo", "%", -1, true, false, false, false
        );

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" % ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<PrimaryNDXConsumer> operationCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if (d < 0) return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] % t2_val[inputs[2].i_of_idx(t1Idx)];
                    else {
                        return t1Idx -> {
                            if (d == 0) {
                                return 1 / t2_val[inputs[2].i_of_idx(t1Idx)];
                            } else {
                                return -(t1_val[inputs[1].i_of_idx(t1Idx)] / Math.pow(t2_val[inputs[2].i_of_idx(t1Idx)], 2));
                            }
                        };
                    }
                };

        Operator operator = new Operator()
            .setADAnalyzer(
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
                    Tsr derivv = (Tsr)call.getAt("derivative");
                    Function mul = Function.Detached.MUL;
                    if (
                        derivv != null
                    ) {
                        return new ADAgent(
                                derivv
                            ).withForward(
                                ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, derivv})
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
                    ( caller, call ) -> null
            ).setRJAgent(
                    ( call, goDeeperWith ) -> null
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
                Operator.class,
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
                                                                        operationCreator.create(call.getTensors(), call.getDerivativeIndex())
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

        DefaultOperatorCreator<TertiaryNDXConsumer> creator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if (d < 0) {
                        return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] % t2_val[inputs[2].i_of_idx(t2Idx)];
                    } else {
                        return (t0Idx, t1Idx, t2Idx) -> {
                            if (d == 0) {
                                return 1 / t2_val[inputs[2].i_of_idx(t2Idx)];
                            } else {
                                return
                                        -(t1_val[inputs[1].i_of_idx(t1Idx)]
                                                /
                                                Math.pow(t2_val[inputs[2].i_of_idx(t2Idx)], 2));
                            }
                        };
                    }
                };

        Broadcast broadcast = new Broadcast()
            .setADAnalyzer(
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
                    Tsr derivv = (Tsr)call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if (
                            derivv != null
                        ) {
                            return new ADAgent(
                                derivv
                            ).withForward(
                                ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, derivv})
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
                Broadcast.class,
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
                                                                        creator.create(call.getTensors(), call.getDerivativeIndex())
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
                                "value = ((int)src1) % ((int)src2);\n",
                                "if(d==0){\n" +
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

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) {
                        return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] % value;
                    } else {
                        if (d == 0) return t1Idx -> 1 / value;
                        else return t1Idx -> -value / Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], 2);
                    }
                };

        Scalarization scalarization = new Scalarization()
            .setADAnalyzer(
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
                    Tsr derivv = (Tsr)call.getAt("derivative");
                    Function mul = Function.Detached.MUL;
                    if (
                        derivv != null
                    ) {
                        return new ADAgent(
                                derivv
                            ).withForward(
                                ( node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, derivv})
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
                    ( caller, call ) -> null
            ).setRJAgent(
                    ( call, goDeeperWith ) -> null
            ).setDrainInstantiation(
                    call -> {
                        Tsr[] tsrs = call.getTensors();
                        int offset = ( tsrs[0] == null ) ? 1 : 0;
                        return new ExecutionCall<>(call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1 + offset]}, -1, OperationType.instance("idy"));
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
                                "output = ((int)input1) % ((int)value);     \n",
                                "if(d==0){                               \n" +
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

        new AbstractOperationType(
                "", ((char) 171) + "%", 3, true, false, false, false
        ) {
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
            }
        };
        new AbstractOperationType(
                "", "%" + ((char) 187), 3, true, false, false, false
        ) {
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
            }
        };
    }



    @Contract(pure = true)
    public static double calculate(double[] inputs, int d, List<Function> src) {
        if ( d < 0 ) {
            double result = src.get(0).call(inputs);
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get(i).call(inputs);
                result %= current;
            }
            return result;
        } else {
            return src.get(0).derive(inputs, d);
        }
    }

    @Contract(pure = true)

    @Override
    public double calculate(double[] inputs, int j, int d, List<Function> src) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src.get(0).call(inputs, j);
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get(i).call(inputs, j);
                result %= current;
            }
            return result;
        } else {
            return src.get(0).derive(inputs, d, j);// j ?
        }
    }





}
