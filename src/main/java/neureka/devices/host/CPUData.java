package neureka.devices.host;

import neureka.devices.AbstractDeviceData;
import neureka.devices.Device;
import neureka.dtype.DataType;

class CPUData<T> extends AbstractDeviceData<T>
{
    public CPUData(Device<?> owner, Object ref, DataType<T> dataType) {
        super(owner, ref, dataType);
    }

    @Override
    protected void _free() {
        // System.getGarbageCollector().collect(_dataRef);
        // ^ This might work in an alternative Universe. :P
    }
}
