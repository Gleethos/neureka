package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  A {@link ForwardADPredicate} lambda checks if forward mode auto differentiation
 *  can be performed for a given {@link ExecutionCall}.
 *  The analyzer returns a boolean truth value.
 */
public interface ForwardADPredicate {

    /**
     *  This method checks if forward mode auto differentiation can
     *  be performed for a given {@link ExecutionCall}.
     *  The analyzer returns a boolean truth value.
     *
     * @param call The {@link ExecutionCall} call for which the forward mode auto differentiation suitability ought to be checked.
     * @return The truth value determining if forward mode auto differentiation can be performed.
     */
    boolean canPerformForwardADFor( ExecutionCall<? extends Device<?>> call );

}
