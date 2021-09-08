package neureka.backend.api.algorithms.api;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 * instantiate new tensors for execution in
 */
public interface DrainInstantiation {

    ExecutionCall<? extends Device<?>> handle(ExecutionCall<? extends Device<?>> call );

}
