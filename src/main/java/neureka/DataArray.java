package neureka;

import neureka.devices.Device;

public interface DataArray {

    Device<?> getDevice();

    Object get();

}
