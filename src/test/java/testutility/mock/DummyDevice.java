package testutility.mock;

import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.devices.AbstractBaseDevice;
import neureka.devices.Device;
import neureka.backend.api.ExecutionCall;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.ndim.config.NDConfiguration;

import java.util.Collection;

public class DummyDevice extends AbstractBaseDevice<Object>
{
    @Override
    public void dispose() { }

    @Override
    public <T> Access<T> access(Tsr<T> tensor) {
        return CPU.get().access(tensor);
    }

    @Override
    public Device<Object> restore( Tsr tensor ) { return this; }

    @Override
    public Device<Object> store( Tsr tensor ) { return this; }

    @Override
    public boolean has( Tsr tensor ) {
        return false;
    }

    @Override
    public Device<Object> free( Tsr tensor ) { return this; }

    @Override
    public Device<Object> approve(ExecutionCall<? extends Device<?>> call ) { return this; }

    @Override
    public Collection<Tsr<Object>> getTensors() { return null; }

    @Override
    public <V> neureka.Data<V> allocate( DataType<V> dataType, NDConfiguration ndc ) { return null; }

    @Override
    public <V> neureka.Data<V> allocate(DataType<V> dataType, int size, V initialValue) { return null; }

    @Override
    public neureka.Data<Object> allocate(Object jvmData, int desiredSize) { return null; }

    @Override
    public Operation optimizedOperationOf(Function function, String name) {
        return null;
    }

    @Override
    public boolean update(OwnerChangeRequest<Tsr<Object>> changeRequest) {
        return true;
    }
}
