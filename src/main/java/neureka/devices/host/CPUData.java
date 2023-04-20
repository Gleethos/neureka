package neureka.devices.host;

import neureka.devices.AbstractDeviceData;
import neureka.devices.Device;
import neureka.dtype.DataType;

import java.util.function.Consumer;

class CPUData<T> extends AbstractDeviceData<T>
{
    private final Consumer<Integer> _usageCountListener;


    CPUData( Device<?> owner, Object ref, DataType<T> dataType, Consumer<Integer> usageCountListener ) {
        super(owner, ref, dataType);
        _usageCountListener = usageCountListener;
    }

    @Override public void incrementUsageCount() {
        super.incrementUsageCount();
        if ( _usageCount > 0 ) _usageCountListener.accept(1);
    }

    @Override public void decrementUsageCount() {
        super.decrementUsageCount();
        if ( _usageCount >= 0 ) _usageCountListener.accept(-1);
    }

    @Override
    protected void _free() {
        // System.getGarbageCollector().collect(_dataRef);
        // ^ This might work in an alternative Universe. :P
    }

}
