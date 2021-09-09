package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface DeviceFinder {

    Device<?> findDeviceFor(ExecutionCall<? extends Device<?>> call );

}
