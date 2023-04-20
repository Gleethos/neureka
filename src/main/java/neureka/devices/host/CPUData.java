package neureka.devices.host;

import neureka.devices.AbstractDeviceData;
import neureka.devices.Device;
import neureka.devices.ReferenceCounter;
import neureka.dtype.DataType;

import java.util.function.Consumer;

class CPUData<T> extends AbstractDeviceData<T>
{
    CPUData( Device<?> owner, Object ref, DataType<T> dataType, Consumer<Integer> usageCountListener ) {
        super(owner, ref, dataType, new ReferenceCounter( changeEvent ->{
            switch ( changeEvent.type() ) {
                case INCREMENT:
                case DECREMENT:
                case FULL_DELETE: usageCountListener.accept(changeEvent.change());
                break;
            }
            if ( changeEvent.currentCount() == 0 ) {
                // System.getGarbageCollector().collect(_dataRef);
                // ^ This might work in an alternative Universe. :P
            }
        }));
    }

}
