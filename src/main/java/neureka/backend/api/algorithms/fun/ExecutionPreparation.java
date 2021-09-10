package neureka.backend.api.algorithms.fun;

import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  An {@link Algorithm} will typically produce a result when executing an {@link ExecutionCall}.
 *  This result must be created somehow.
 *  A {@link ExecutionPreparation} implementation instance will do just that...
 */
public interface ExecutionPreparation {

    ExecutionCall<? extends Device<?>> prepare(ExecutionCall<? extends Device<?>> call );

}
