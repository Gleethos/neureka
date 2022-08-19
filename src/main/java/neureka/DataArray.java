package neureka;

import neureka.devices.Device;

public interface DataArray {

    /**
     * @return The owner of this data array.
     */
    Device<?> owner();

    /**
     * @return The raw underlying data of this wrapper.
     */
    Object get();

}
