package neureka.backend.api.algorithms.fun;

import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  A {@link BackwardADChecker} lambda checks if this
 *  {@link Algorithm} can perform backward AD for a given {@link ExecutionCall}.
 *  The analyzer returns a boolean truth value.
 */
public interface BackwardADChecker {

    boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call );

}
