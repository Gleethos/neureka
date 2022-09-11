package neureka.backend.api.ini;

import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Operation;
import neureka.devices.Device;

import java.util.function.Function;

public interface ImplementationReceiver
{
    <D extends Device<?>> void accept(
            Class<? extends Operation> operationType,
            Class<? extends DeviceAlgorithm> algorithmType,
            Class<? extends D> deviceType,
            Function<LoadingContext, ImplementationFor<D>> function
    );
}
