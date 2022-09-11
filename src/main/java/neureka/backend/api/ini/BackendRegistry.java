package neureka.backend.api.ini;

import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Operation;
import neureka.devices.Device;

import java.util.function.Function;

public final class BackendRegistry
{
    private final ImplementationReceiver _receiver;

    public static BackendRegistry of( ImplementationReceiver receiver )
    {
        return new BackendRegistry( receiver );
    }

    private BackendRegistry(ImplementationReceiver receiver) {
        _receiver = receiver;
    }


    public <D extends Device<?>> RegisterForDevice<D> forDevice( Class<? extends D> deviceType )
    {
        return new RegisterForDevice<D>() {
            @Override
            public <A extends DeviceAlgorithm> RegisterForDevice<D> set(
                Class<? extends Operation> operationType,
                Class<? extends A> algorithmType,
                Function<LoadingContext, ImplementationFor<D>> function
            ) {
                _receiver.accept( operationType, algorithmType, deviceType, function );
                return this;
            }

            @Override
            public RegisterForOperation<D> andOperation(Class<? extends Operation> operationType) {
                return new RegisterForOperation<D>() {
                    @Override
                    public RegisterForOperation<D> set(
                        Class<? extends DeviceAlgorithm> algorithmType,
                        Function<LoadingContext, ImplementationFor<D>> function
                    ) {
                        _receiver.accept( operationType, algorithmType, deviceType, function );
                        return this;
                    }
                };
            }
        };
    }

}
