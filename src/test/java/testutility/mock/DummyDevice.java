package testutility.mock;

import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.devices.AbstractBaseDevice;
import neureka.devices.Device;
import neureka.backend.api.ExecutionCall;

import java.util.Collection;

public class DummyDevice extends AbstractBaseDevice<Object>
{
    @Override
    public void dispose() { }

    @Override
    public <T> Access<T> access(Tsr<T> tensor) {
        return null;
    }

    @Override
    public Device<Object> restore( Tsr tensor ) { return this; }

    @Override
    public Device<Object> store( Tsr tensor ) { return this; }

    @Override
    public Device<Object> store( Tsr tensor, Tsr parent ) { return this; }

    @Override
    public boolean has(Tsr tensor) {
        return false;
    }

    @Override
    public Device<Object> free(Tsr tensor) { return this; }

    @Override
    public Device<Object> cleaning(Tsr tensor, Runnable action) { return null; }

    @Override
    public <T extends Object> Device<Object> swap( Tsr<T> former, Tsr<T> replacement ) { return this; }

    @Override
    public Device<Object> approve(ExecutionCall<? extends Device<?>> call ) { return this; }

    @Override
    public <T> Device<Object> updateNDConf( Tsr<T> tensor ) {
        return this;
    }

    @Override
    public Collection<Tsr<Object>> getTensors() { return null; }

    @Override
    public Operation optimizedOperationOf(Function function, String name) {
        return null;
    }

    @Override
    public boolean update(OwnerChangeRequest<Tsr<Object>> changeRequest) {
        return true;
    }
}
