package neureka.devices.host;

import neureka.devices.AbstractBaseDevice;
import neureka.devices.AbstractDeviceData;
import neureka.devices.ReferenceCounter;
import neureka.dtype.DataType;

import java.util.function.Consumer;

class CPUData<T> extends AbstractDeviceData<T>
{
    CPUData( AbstractBaseDevice<?> owner, Object ref, DataType<T> dataType ) {
        super(owner, ref, dataType, ()->{
            // This lambda exists to do some cleanup when the data is no longer needed. So... what to do?
            // System.getGarbageCollector().collect(ref);
            // ^ This might work in an alternative Universe. :P
        });
    }

}
