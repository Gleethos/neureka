package neureka.ndim.iterators.types;

import neureka.ndim.config.types.virtual.VirtualNDConfiguration;
import neureka.ndim.iterators.NDIterator;

public final class VirtualNDIterator implements NDIterator
{
    private final VirtualNDConfiguration _conf;

    public VirtualNDIterator( VirtualNDConfiguration ndc ) {
        _conf = ndc;
    }

    @Override
    public int shape( int i ) {
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
        throw new IllegalStateException(
                "Virtual ND-iterator do not keep track of the iteration index!"
        );
    }

    @Override
    public int[] get() {
        throw new IllegalStateException(
                "Virtual ND-iterator do not keep track of the iteration index!"
        );
    }

    @Override
    public void set(int axis, int position) {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public void set( int[] indices ) {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public int rank() {
        return _conf.rank();
    }
}
