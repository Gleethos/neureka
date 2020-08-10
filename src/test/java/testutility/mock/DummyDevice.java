package testutility.mock;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.calculus.environment.OperationType;

import java.util.Collection;

public class DummyDevice implements Device
{
    @Override
    public void dispose() {

    }

    @Override
    public Device get(Tsr tensor) {
        return this;
    }

    @Override
    public Device add(Tsr tensor) {
        return this;
    }

    @Override
    public Device add(Tsr tensor, Tsr parent) {
        return this;
    }

    @Override
    public boolean has(Tsr tensor) {
        return false;
    }

    @Override
    public Device rmv(Tsr tensor) {
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
    public Device execute(Tsr[] tsrs, OperationType type, int d) {
        return this;
    }

    @Override
    public double[] value64f( Tsr tensor ) {
        return new double[0];
    }

    @Override
    public float[] value32f( Tsr tensor ) {
        return new float[0];
    }

    @Override
    public double value64f(Tsr tensor, int index) {
        return 0;
    }

    @Override
    public float value32f(Tsr tensor, int index) {
        return 0;
    }

    @Override
    public Collection<Tsr> tensors() {
        return null;
    }

    @Override
    public void update(Tsr oldOwner, Tsr newOwner) {

    }
}
