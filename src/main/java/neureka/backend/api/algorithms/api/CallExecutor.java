package neureka.backend.api.algorithms.api;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface CallExecutor {

    Tsr<?> execute(ExecutionCall<? extends Device<?>> call );

}
