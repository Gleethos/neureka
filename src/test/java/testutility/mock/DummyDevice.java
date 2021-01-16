package testutility.mock;

import neureka.Tsr;
import neureka.devices.AbstractBaseDevice;
import neureka.devices.Device;
import neureka.backend.api.ExecutionCall;

import java.util.Collection;

public class DummyDevice extends AbstractBaseDevice<Object>
{
    @Override
    public void dispose() {

    }

    @Override
    public Device restore(Tsr tensor) {
        return this;
    }

    @Override
    public Device store(Tsr tensor) {
        return this;
    }

    @Override
    public Device store(Tsr tensor, Tsr parent) {
        return this;
    }

    @Override
    public boolean has(Tsr tensor) {
        return false;
    }

    @Override
    public Device free(Tsr tensor) {
        return this;
    }

    @Override
    public Device cleaning(Tsr tensor, Runnable action) {
        return null;
    }

    @Override
    public Device overwrite64(Tsr tensor, double[] value) {
        return this;
    }

    @Override
    public Device overwrite32(Tsr tensor, float[] value) {
        return this;
    }

    @Override
    public Device swap(Tsr former, Tsr replacement) {
        return this;
    }

    @Override
    public Device execute( ExecutionCall call ) {
        return this;
    }

    @Override
    public Object valueFor(Tsr<Object> tensor) {
        return null;
    }

    @Override
    public Object valueFor(Tsr<Object> tensor, int index) {
        return null;
    }

    @Override
    public Collection<Tsr<Object>> getTensors() {
        return null;
    }

    @Override
    public void update(Tsr oldOwner, Tsr newOwner) {

    }
}
