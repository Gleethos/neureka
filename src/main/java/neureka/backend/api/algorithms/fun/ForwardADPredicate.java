package neureka.backend.api.algorithms.fun;

import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  A {@link ForwardADPredicate} lambda checks if this
 *  {@link Algorithm} can perform forward AD for a given {@link ExecutionCall}.
 *  The analyser return a boolean truth value.
 */
public interface ForwardADPredicate {

    boolean canPerformForwardADFor( ExecutionCall<? extends Device<?>> call );

}
