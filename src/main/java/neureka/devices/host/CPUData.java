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
            // System.getGarbageCollector().collect(_dataRef);
            // ^ This might work in an alternative Universe. :P
        });
    }

}
