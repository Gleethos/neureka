package neureka.ndim.iterator.types.simple;


import neureka.ndim.config.types.simple.Simple1DConfiguration;
import neureka.ndim.iterator.NDIterator;

public final class Simple1DCIterator extends Simple1DConfiguration implements NDIterator
{
    private int _d1;

    public Simple1DCIterator(Simple1DConfiguration ndc ) {
        super( ndc.shape( 0 ), ndc.translation( 0 ));
    }


    /** {@inheritDoc} */
    @Override public final void increment() { _d1++; }

    /** {@inheritDoc} */
    @Override public final void decrement() { _d1--; }

    /** {@inheritDoc} */
    @Override public final int i() { return this.indexOfIndices(_d1); }

    /** {@inheritDoc} */
    @Override public final int get( int axis ) { return _d1; }

    /** {@inheritDoc} */
    @Override public final int[] get() { return new int[]{_d1}; }

    /** {@inheritDoc} */
    @Override public final void set( int axis, int position ) { _d1 = position; }

    /** {@inheritDoc} */
    @Override public final void set( int[] indices) { _d1 = indices[0]; }

}
