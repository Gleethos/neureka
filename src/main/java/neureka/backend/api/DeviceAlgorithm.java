package neureka.backend.api;

import neureka.devices.Device;

/**
 *  A {@link DeviceAlgorithm} is an advanced form of {@link Algorithm} which
 *  delegates the execution to implementations of {@link ImplementationFor} specific {@link Device} types.
 *
 * @param <C>
 */
public interface DeviceAlgorithm<C extends DeviceAlgorithm<C>> extends Algorithm {

    /**
     * Implementations of the {@link DeviceAlgorithm} interface ought to express a compositional design pattern. <br>
     * This means that concrete implementations of an algorithm for a device are not extending
     * an Algorithm, they are components of it instead. <br>
     * These components can be stored on an Algorithm by passing
     * a Device class as key and an ImplementationFor instance as value.
     *
     * @param deviceClass    The class of the {@link Device} for which an implementation should be set.
     * @param implementation The {@link ImplementationFor} the provided {@link Device} type.
     * @param <D>            The type parameter of the {@link Device} type for which
     *                       an implementation should be set in this {@link Device}.
     * @param <I>            The type of the {@link ImplementationFor} the provided {@link Device} type.
     * @return This very {@link Algorithm} instance to allow for method chaining.
     */
    <D extends Device<?>, I extends ImplementationFor<D>> C setImplementationFor(Class<D> deviceClass, I implementation );

    /**
     * An {@link ImplementationFor} a specific {@link Device} can be accessed by passing the class of
     * the {@link Device} for which an implementation should be returned.
     * An Algorithm instance ought to contain a collection of these {@link Device} specific
     * implementations...
     *
     * @param deviceClass The class of the device for which the stored algorithm implementation should be returned.
     * @param <D>         The type parameter which has to be a class extending the Device interface.
     * @return The implementation for the passed device type class.
     */
    <D extends Device<?>> ImplementationFor<D> getImplementationFor( Class<D> deviceClass );

    /**
     * An {@link ImplementationFor} a specific {@link Device} can be accessed by passing the
     * the {@link Device} for which an implementation should be returned.
     * An Algorithm instance ought to contain a collection of these {@link Device} specific
     * implementations...
     *
     * @param device The device for which the stored algorithm implementation should be returned.
     * @param <D>    type parameter which has to be a class extending the Device interface.
     * @return The implementation for the passed device type class.
     */
    default <D extends Device<?>> ImplementationFor<D> getImplementationFor( D device ) {
        return (ImplementationFor<D>) getImplementationFor(device.getClass());
    }

}
