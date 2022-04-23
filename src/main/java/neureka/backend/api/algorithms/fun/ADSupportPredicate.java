package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;


/**
 *  A {@link ADSupportPredicate} lambda checks which auto differentiation mode
 *  can be performed for a given {@link ExecutionCall}.
 *  The analyzer returns a {@link AutoDiff} enum instance.
 */
public interface ADSupportPredicate {

    /**
     *  Implementations of this ought to check which auto differentiation mode
     *  can be performed for a given {@link ExecutionCall}.
     * @param call The {@link ExecutionCall} which should be checked.
     * @return A {@link AutoDiff} enum instance describing what kind of differentiation can be performed.
     */
    AutoDiff autoDiffModeFrom(ExecutionCall<? extends Device<?>> call );

}
