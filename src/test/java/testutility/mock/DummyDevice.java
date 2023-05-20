package testutility.mock;

import neureka.Data;
import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.devices.AbstractBaseDevice;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.math.Function;
import neureka.ndim.config.NDConfiguration;

public class DummyDevice extends AbstractBaseDevice<Object>
{
    @Override
    public void dispose() { }

    @Override
    public <T> Access<T> access(Tensor<T> tensor) {
        return CPU.get().access(tensor);
    }

    @Override
    public Device<Object> restore( Tensor tensor ) { return this; }

    @Override
    public Device<Object> store( Tensor tensor ) { return this; }

    @Override
    public boolean has( Tensor tensor ) {
        return false;
    }

    @Override
    public Device<Object> free( Tensor tensor ) { return this; }

    @Override
    public Device<Object> approve( ExecutionCall<? extends Device<?>> call ) { return this; }

    @Override
    public <V> Data<V> allocate(DataType<V> dataType, NDConfiguration ndc ) { return null; }

    @Override
    public <V> Data<V> allocateFromOne(DataType<V> dataType, NDConfiguration ndc, V initialValue ) { return null; }

    @Override
    public <T> Data<T> allocateFromAll(DataType<T> dataType, NDConfiguration ndc, Object jvmData ) { return null; }

    @Override
    public Operation optimizedOperationOf( Function function, String name ) {
        return null;
    }

    @Override
    public boolean update( OwnerChangeRequest<Tensor<Object>> changeRequest ) {
        return true;
    }
}
