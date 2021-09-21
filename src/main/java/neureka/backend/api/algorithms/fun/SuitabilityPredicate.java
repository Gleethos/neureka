package neureka.backend.api.algorithms.fun;

import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.devices.Device;

/**
 *  The {@link SuitabilityPredicate} checks if a given instance of an {@link ExecutionCall} is
 *  suitable to be executed in {@link neureka.backend.api.ImplementationFor}
 *  residing in this {@link Algorithm} as components.
 *  It can be implemented as s simple lambda.
 */
public interface SuitabilityPredicate {

    /**
     * When an {@link ExecutionCall} instance has been formed then it will be routed by <br>
     * the given {@link Operation} instance to their components, namely : <br>
     * {@link Algorithm} instances ! <br>
     *
     * The ability to decide which algorithm is suitable for a given {@link ExecutionCall} instance <br>
     * is being granted by implementations of the following method. <br>
     * It <b>returns a float representing the suitability of a given call</b>. <br>
     * The float is expected to be between 0 and 1, where 0 means <br>
     * that the implementation is not suitable at all and 1 means that <br>
     * that it fits the call best! <br>
     *
     * @param call The {@link ExecutionCall} whose suitability for execution on this {@link Algorithm} ought to be determined.
     * @return The suitability degree expressed by a float value between 0 and 1, where 0 means not suitable and 1 means suitable.
     */
    float isSuitableFor(ExecutionCall<? extends Device<?>> call );

}
