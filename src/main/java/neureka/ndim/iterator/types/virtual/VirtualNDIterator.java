package neureka.ndim.iterator.types.virtual;

import neureka.ndim.config.types.views.virtual.VirtualNDConfiguration;
import neureka.ndim.iterator.NDIterator;

public final class VirtualNDIterator implements NDIterator
{
    private final VirtualNDConfiguration _conf;

    public VirtualNDIterator( VirtualNDConfiguration ndc ) { _conf = ndc; }

    @Override
    public final int shape( int i ) { return _conf.shape(i); }

    @Override
    public final int[] shape() { return _conf.shape(); }

    @Override
    public final void increment() {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public final void decrement() {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public final int i() {
        return 0;
    }

    @Override
    public final int get(int axis) {
        throw new IllegalStateException(
                "A virtual ND-iterator does not keep track of the iteration index! " +
                "You cannot use this type of iterator when the data access pattern of your algorithm " +
                "relies on this type of information."
        );
    }

    @Override
    public final int[] get() {
        throw new IllegalStateException(
                "A virtual ND-iterator does not keep track of the iteration index! " +
                "You cannot use this type of iterator when the data access pattern of your algorithm " +
                "relies on this type of information."
        );
    }

    @Override
    public final void set(int axis, int position) {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public final void set( int[] indices ) {
        // It's virtual and therefore does nothing :)
    }

    @Override
    public final int rank() { return _conf.rank(); }
}
