package neureka.devices;

import neureka.common.utility.LogUtil;
import neureka.dtype.DataType;
import neureka.ndim.config.NDConfiguration;

public abstract class AbstractDeviceData<T> implements DeviceData<T>
{
    protected final AbstractBaseDevice<?> _owner;
    protected final Object _dataRef;
    protected final DataType<T> _dataType;
    protected final ReferenceCounter _refCounter;


    public AbstractDeviceData(
        AbstractBaseDevice<?> owner,
        Object ref,
        DataType<T> dataType,
        Runnable cleanup
    ) {
        LogUtil.nullArgCheck(owner, "owner", Device.class);
        LogUtil.nullArgCheck(dataType, "dataType", DataType.class);
        LogUtil.nullArgCheck(cleanup, "cleanup", Runnable.class);
        ReferenceCounter counter = new ReferenceCounter( changeEvent ->{
                                        owner._numberOfTensors += changeEvent.change();
                                        if ( changeEvent.currentCount() == 0 )
                                            cleanup.run();
                                    });

        _owner = owner;
        _dataRef = ref;
        _dataType = dataType;
        _refCounter = counter;
        DeviceCleaner.INSTANCE.register( this, ()->{
            if ( counter.count() > 0 )
                owner._numberOfDataObjects--;

            counter.fullDelete();
        });
    }

    @Override public final Device<T> owner() { return (Device<T>) _owner; }

    @Override public final Object getOrNull() { return _dataRef; }

    @Override public final DataType<T> dataType() { return _dataType; }

    @Override public final void incrementUsageCount() {
        if ( _refCounter.count() == 0 ) _owner._numberOfDataObjects++;
        _refCounter.increment();
    }

    @Override public final void decrementUsageCount() {
        if ( _refCounter.count() == 1 ) _owner._numberOfDataObjects--;
        _refCounter.decrement();
    }

    @Override public final int usages() { return _refCounter.count(); }
}
