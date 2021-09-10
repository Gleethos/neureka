package neureka.backend.api.algorithms.fun;

import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.devices.Device;

/**
 *  This {@link neureka.backend.api.algorithms.fun.ADAgentSupplier} will supply
 *  {@link ADAgent} instances which can perform backward and forward auto differentiation.
 */
public interface ADAgentSupplier {

    ADAgent supplyADAgentFor(
            Function f,
            ExecutionCall<? extends Device<?>> call,
            boolean forward
    );

}
