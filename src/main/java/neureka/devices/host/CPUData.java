package neureka.devices.host;

import neureka.devices.AbstractDeviceData;
import neureka.devices.Device;
import neureka.dtype.DataType;

class CPUData<T> extends AbstractDeviceData<T> {
    public CPUData(Device<?> owner, Object ref, DataType<T> dataType) {
        super(owner, ref, dataType);
    }
}
