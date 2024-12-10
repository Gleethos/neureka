package neureka;

import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;

final class NoOpData implements Data<Void>
{
    static final NoOpData INSTANCE = new NoOpData();

    private NoOpData() {}

    @Override
    public Device<Void> owner() {
        return (Device) CPU.get();
    }

    @Override
    public Object getOrNull() {
        return null;
    }

    @Override
    public DataType<Void> dataType() {
        return DataType.of(Void.class);
    }

    @Override
    public int usages() {
        return 0;
    }
}
