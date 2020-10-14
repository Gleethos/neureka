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

public class Subtraction extends AbstractOperationType
{

    public Subtraction()
    {
        super(
                "subtract", "-", -1, true, false, true, false
        );

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
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
            OperationType type = call.getType();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if (d < 0) {
                    Tsr[] reduction = new Tsr[]{tsrs[0], tsrs[1], tsrs[2]};
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<Device>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];

                    reduction = Utility._offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<Device>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];
                } else {
                    tsrs[0] = Tsr.Create.newTsrLike(tsrs[1]).setValue((d==0)?1.0f:-1.0f);
                }
                return alternative;
            } else {
                return alternative;
            }
        };

        //_____________________
        // DEFAULT OPERATION :

        DefaultOperatorCreator<PrimaryNDXConsumer> operationCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if ( d < 0 ) {
                        return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] - t2_val[inputs[2].i_of_idx(t1Idx)];
                    } else return t1Idx -> ( d == 0 ) ? 1.0 : -1.0;
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
                                "output = input1 - input2;  \n",
                                "if(d==0){                 \n" +//drn and src2 switch:
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

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarOperatorCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if ( d < 0 ) return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] - value;
                    else if ( d == 0 ) return t1Idx -> 1; else return t1Idx -> -1;
                };

        Scalarization scalarization = new Scalarization()
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> true )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getAt("derivative");
                        Function mul = Function.Detached.MUL;
                        if ( ctxDerivative != null ) {
                            return new ADAgent( ctxDerivative )
                                    .withForward( ( node, forwardDerivative ) -> mul.call( new Tsr[]{forwardDerivative, ctxDerivative} ) )
                                    .withBackward( ( node, backwardError ) -> mul.call( new Tsr[]{backwardError, ctxDerivative} ) );
                        }
                        Tsr<?> localDerivative = f.derive(call.getTensors(), call.getDerivativeIndex());
                        return new ADAgent( localDerivative )
                                .withForward( (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, localDerivative}) )
                                .withBackward( (node, backwardError) -> mul.call(new Tsr[]{backwardError, localDerivative}) );
                    }
                )
                .setCallHock( (caller, call) -> null )
                .setRJAgent( rja )
                .setDrainInstantiation(
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
                scalarization.setExecutor (
                        HostExecutor.class,
                        new HostExecutor (
                                call -> {
                                    int offset = (call.getTensor(2).isVirtual() || call.getTensor(2).size() == 1) ? 1 : 0;
                                    double value = call.getTensor(1+offset).value64(0);
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
                                "output = input1 - value;\n",
                                "if(d==0){     \n" +//drn and src2 switch:
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

        setImplementation (
                Broadcast.class,
                new Broadcast()
                        .setBackwardADAnalyzer( call -> true )
                        .setForwardADAnalyzer( call -> true )
                        .setADAgentSupplier(
                            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                            {
                                Tsr<?> ctxDerivative = (Tsr<?>)call.getAt("derivative");
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
                        )
                        .setCallHock( (caller, call) -> null )
                        .setRJAgent( rja )
                        .setDrainInstantiation(
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
                        )
                        // add _creator
                    );

        //______________________
        // RELATED OPERATIONS :

        new AbstractOperationType(
                "", ((char) 171) + "-", 3, true, false, false, false
        ) {
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
            }
        };
        new AbstractOperationType(
                "", "-" + ((char) 187), 3, true, false, false, false
        ) {
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
            }
        };

        // Convolution:


        new AbstractOperationType(
                "", "s", 2, true, false, false, false
        ) {
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
            }
        }.setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
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
            public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
            }
        };
        new AbstractOperationType(
                "", "s" + ((char) 187), 3, true, false, false, false
        ) {
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
            }
        };


    }


    @Contract(pure = true)

    @Override
    public double calculate(double[] inputs, int j, int d, List<Function> src) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src.get(0).call(inputs, j);
            for ( int Vi = 1; Vi < src.size(); Vi++ ) {
                final double current = src.get(Vi).call(inputs, j);
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.size(); ++i ) {
                if (i == 0) {
                    derivative += src.get(i).derive(inputs, d, j);
                } else {
                    derivative -= src.get(i).derive(inputs, d, j);
                }
            }
            return derivative;
        }
    }

    @Contract(pure = true)
    public static double calculate(double[] inputs, int d, List<Function> src) {
        if ( d < 0 ) {
            double result = src.get(0).call(inputs);
            for ( int i = 1; i < src.size(); i++ ) {
                final double current = src.get(i).call(inputs);
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.size(); ++i ) {
                if ( i == 0 ) {
                    derivative += src.get(i).derive(inputs, d);
                } else {
                    derivative -= src.get(i).derive(inputs, d);
                }
            }
            return derivative;
        }
    }



}
