package neureka.backend.api.algorithms.fun;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  The {@link neureka.backend.api.algorithms.fun.DeviceFinder} finds
 *  a {@link Device} instance which fits the contents of a given {@link ExecutionCall} instance.
 *  The finder is supposed to find a {@link Device} which can be most easily shared
 *  by the {@link Tsr} instances within the {@link ExecutionCall} that is being received by the finder.
 *  This can be implemented as a simple lambda.
 */
public interface DeviceFinder {

    Device<?> findDeviceFor(ExecutionCall<? extends Device<?>> call );

}
