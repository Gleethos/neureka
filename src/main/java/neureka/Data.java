package neureka;

import neureka.devices.Device;

/**
 *  A wrapper type for the raw data array of a tensor/nd-array,
 *  which is provided by implementations of the {@link Device} interface.
 *  This is used to interface with the raw data as well as check where it comes from.
 *
 * @param <V> The type of the data array.
 */
public interface Data<V> {

    /**
     * @return The owner of this data array wrapper (the device which allocated the memory).
     */
    Device<V> owner();

    /**
     * @return The raw underlying data of this wrapper.
     */
    Object get();

}
