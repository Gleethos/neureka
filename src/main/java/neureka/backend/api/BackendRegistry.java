package neureka.backend.api;

import neureka.devices.Device;

import java.util.function.Function;

public interface BackendRegistry
{
    <D extends Device<?>> void register(
            Class<? extends Operation> operationType,
            Class<? extends DeviceAlgorithm> algorithmType,
            Class<? extends D> deviceType,
            Function<Context, ImplementationFor<D>> function
    );

    interface RegisterForOperation<D extends Device<?>>
    {
        RegisterForOperation<D> set(
                Class<? extends DeviceAlgorithm> algorithmType,
                Function<Context, ImplementationFor<D>> function
        );

    }

    interface RegisterForDevice<D extends Device<?>>
    {
        <A extends DeviceAlgorithm> RegisterForDevice<D> set(
            Class<? extends Operation> operationType,
            Class<? extends A> algorithmType,
            Function<Context, ImplementationFor<D>> function
        );

        RegisterForOperation<D> andOperation(Class<? extends Operation> operationType );
    }

    default <D extends Device<?>> RegisterForDevice<D> forDevice( Class<? extends D> deviceType )
    {
        return new RegisterForDevice<D>() {
            @Override
            public <A extends DeviceAlgorithm> RegisterForDevice<D> set(
                Class<? extends Operation> operationType,
                Class<? extends A> algorithmType,
                Function<Context, ImplementationFor<D>> function
            ) {
                register( operationType, algorithmType, deviceType, function );
                return this;
            }

            @Override
            public RegisterForOperation<D> andOperation(Class<? extends Operation> operationType) {
                return new RegisterForOperation<D>() {
                    @Override
                    public RegisterForOperation<D> set(
                        Class<? extends DeviceAlgorithm> algorithmType,
                        Function<Context, ImplementationFor<D>> function
                    ) {
                        register( operationType, algorithmType, deviceType, function );
                        return this;
                    }
                };
            }
        };
    }

    interface Context {
        String getIdentifier();
    }
}
