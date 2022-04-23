package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;


/**
 *  A {@link ADSupportPredicate} lambda checks if which auto differentiation mode
 *  can be performed for a given {@link ExecutionCall}.
 *  The analyzer returns a {@link AutoDiff} enum instance.
 */
public interface ADSupportPredicate {

    AutoDiff autoDiffModeFrom(ExecutionCall<? extends Device<?>> call );

}
