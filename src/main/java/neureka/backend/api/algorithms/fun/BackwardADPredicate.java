package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  A {@link BackwardADPredicate} lambda checks if backward mode auto differentiation
 *  (also known as back-propagation) can be performed for a given {@link ExecutionCall}.
 *  The analyzer returns a boolean truth value.
 */
public interface BackwardADPredicate {

    /**
     *  This method checks if backward mode auto differentiation (also known as back-propagation)
     *  can be performed for a given {@link ExecutionCall}.
     *  The analyzer returns a boolean truth value.
     *
     * @param call The {@link ExecutionCall} call for which the backward mode auto differentiation suitability ought to be checked.
     * @return The truth value determining if backward mode auto differentiation can be performed.
     */
    boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call );

}
