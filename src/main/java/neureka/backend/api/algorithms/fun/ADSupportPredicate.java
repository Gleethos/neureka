package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;


/**
 *  A {@link ADSupportPredicate} lambda checks if which auto differentiation mode
 *  can be performed for a given {@link ExecutionCall}.
 *  The analyzer returns a {@link ADMode} enum instance.
 */
public interface ADSupportPredicate {

    enum ADMode { FORWARD_ONLY, BACKWARD_ONLY, NO_AD, FORWARD_AND_BACKWARD }

    ADMode autogradModeFrom( ExecutionCall<? extends Device<?>> call );

}
