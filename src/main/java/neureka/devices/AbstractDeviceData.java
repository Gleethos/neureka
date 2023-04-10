package neureka.devices;

import neureka.common.utility.LogUtil;
import neureka.dtype.DataType;
import neureka.ndim.config.NDConfiguration;

public abstract class AbstractDeviceData<T> implements DeviceData<T>
{
    protected final Device<?> _owner;
    protected final Object _dataRef;
    protected final DataType<T> _dataType;

    public AbstractDeviceData(
            Device<?> owner,
            Object ref,
            DataType<T> dataType
    ) {
        LogUtil.nullArgCheck(owner, "owner", Device.class);
        LogUtil.nullArgCheck(dataType, "dataType", DataType.class);
        _owner = owner;
        _dataRef = ref;
        _dataType = dataType;
    }

    @Override public final Device<T> owner() { return (Device<T>) _owner; }

    @Override public final Object getOrNull() { return _dataRef; }

    @Override public final DataType<T> dataType() { return _dataType; }

    @Override public DeviceData<T> withNDConf(NDConfiguration ndc) { return this; }
}
