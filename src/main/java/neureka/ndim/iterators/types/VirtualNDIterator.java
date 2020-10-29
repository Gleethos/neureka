package neureka.ndim.iterators.types;

import neureka.ndim.config.NDIterator;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;

public class VirtualNDIterator implements NDIterator
{
    private final VirtualNDConfiguration _conf;

    public VirtualNDIterator( VirtualNDConfiguration ndc ) {
        _conf = ndc;
    }

    @Override
    public int shape(int i) {
        return _conf.shape(i);
    }

    @Override
    public int[] shape() {
        return _conf.shape();
    }

    @Override
    public void increment() {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public void decrement() {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public int i() {
        return 0;
    }

    @Override
    public int get(int axis) {
        return 0;
    }

    @Override
    public int[] get() {
        return new int[_conf.rank()];
    }

    @Override
    public void set(int axis, int position) {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public void set(int[] idx) {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public int rank() {
        return _conf.rank();
    }
}
