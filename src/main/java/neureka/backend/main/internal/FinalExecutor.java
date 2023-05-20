package neureka.backend.main.internal;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface FinalExecutor {

    Tensor<?> execute(ExecutionCall<? extends Device<?>> call );

}
