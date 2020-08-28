package neureka.calculus.environment.operations.other;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.Activation;

public class CopyLeft extends OperationType {

    public CopyLeft(){

        super(
                "idy", "<", 2,true, false, false, false
        );

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" <- ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        Activation activation = new Activation()
        .setADAnalyzer(
                call -> {
                    Tsr last = null;
                    for ( Tsr t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        ).setADAgentCreator(
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
                    call.getTensor(offset).incrementVersionBecauseOf(call);
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                }
        );

        setImplementation(Activation.class,
                activation.setExecutor(
                        HostExecutor.class,
                        new HostExecutor(
                                call -> {
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecutor(HostExecutor.class)
                                            .getExecution().call(call);
                                },
                                3
                        )
                ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    OperationType.instance("idy")
                                            .getImplementation(Activation.class)
                                            .getExecutor(CLExecutor.class)
                                            .getExecution().call(call);
                                },
                                3
                        )
                )
        );
    }



}
