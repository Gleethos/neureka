package neureka;

import neureka.devices.Device;

public interface DataArray<V> {

    /**
     * @return The owner of this data array (the device which allocated the memory).
     */
    Device<V> owner();

    /**
     * @return The raw underlying data of this wrapper.
     */
    Object get();

}
