package neureka.backend.api.ini;

import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.ImplementationFor;
import neureka.devices.Device;

import java.util.function.Function;

public interface ReceiveForOperation<D extends Device<?>> {
    ReceiveForOperation<D> set(
            Class<? extends DeviceAlgorithm> algorithmType,
            Function<LoadingContext, ImplementationFor<D>> function
    );

}
