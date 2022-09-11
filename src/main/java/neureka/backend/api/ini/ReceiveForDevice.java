package neureka.backend.api.ini;

import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Operation;
import neureka.devices.Device;

import java.util.function.Function;

public interface ReceiveForDevice<D extends Device<?>> {
    <A extends DeviceAlgorithm> ReceiveForDevice<D> set(
            Class<? extends Operation> operationType,
            Class<? extends A> algorithmType,
            Function<LoadingContext, ImplementationFor<D>> function
    );

    ReceiveForOperation<D> forOperation( Class<? extends Operation> operationType );
}
