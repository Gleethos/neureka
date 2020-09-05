package neureka.calculus.backend.operations.other;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.implementations.Activation;
import neureka.calculus.backend.implementations.Scalarization;

import java.util.List;

public class CopyLeft extends AbstractOperationType {

    public CopyLeft(){

        super(
                "left_inline", "<", 2,true, false, false, false
        );

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
                        if ( i < children.size() - 1 ) reconstructed.append(" <- ");
                    }
                    return "(" + reconstructed + ")";
                }
        );



        Scalarization scalarization = new Scalarization()
                .setSuitabilityChecker(
                        call ->
                        {
                            if ( call.getTensor(1).isVirtual() || call.getTensor(1).size() == 1 ) {
                                return true;
                            } else return false;
                        }
                ).setADAnalyzer(
                        call ->
                        {
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
                ).setCallHock( ( caller, call ) -> null )
                .setRJAgent( ( call, goDeeperWith ) -> null )
                .setDrainInstantiation(
                        call ->
                        {
                            Tsr[] tsrs = call.getTensors();
                            int offset = ( tsrs[0] == null ) ? 1 : 0;
                            call.getTensor(offset).incrementVersionBecauseOf(call);
                            return new ExecutionCall(
                                    call.getDevice(),
                                    new Tsr[]{tsrs[offset], tsrs[1+offset]},
                                    -1,
                                    this
                            );
                        }
                );

        ScalarOperatorCreator<PrimaryNDXConsumer> scalarCreator =
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] = value;
                    else return null;
                };

        setImplementation(
                Scalarization.class,
                scalarization.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                {
                                    double value = call.getTensor(1).value64(0);
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
                                    Tsr t = call.getTensor(0);
                                    int gwz = t.size();
                                    call.getDevice().getKernel(call)
                                            .pass(t)
                                            .pass(t)
                                            .pass(call.getTensor(1).value32(0))
                                            .pass(t.rank())
                                            .pass(call.getDerivativeIndex())
                                            .call(gwz);
                                },
                                3,
                                scalarization.getKernelSource(), // kernelSource
                                "output = value;\n",
                                "output = value;\n",
                                this // OperationType
                        )
                )
        );



        Activation activation = new Activation()
        .setADAnalyzer(
                call ->
                {
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
                call ->
                {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[0] == null ) ? 1 : 0;
                    call.getTensor(offset).incrementVersionBecauseOf(call);
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                }
        );

        setImplementation(
                Activation.class,
                activation
                    .setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call ->
                                {
                                    call.getTensor(0).setIsVirtual(false);
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecutor(HostExecutor.class)
                                            .getExecution().run(call);
                                },
                                3
                        )
                    )
                    .setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    call.getTensor(0).setIsVirtual(false);
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecutor(CLExecutor.class)
                                            .getExecution().run(call);
                                },
                                3
                        )
                )
        );
    }


    @Override
    public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
    }
}
