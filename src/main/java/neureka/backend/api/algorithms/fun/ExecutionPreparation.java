package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 * instantiate new tensors for execution in
 */
public interface ExecutionPreparation {

    ExecutionCall<? extends Device<?>> prepare(ExecutionCall<? extends Device<?>> call );

}
