package neureka.fluent.slicing.states;

import neureka.Tsr;

public interface StepsOrAxisOrGetTsr<V> extends StepsOrAxisOrGet<V>, AxisOrGetTsr<V>
{
    /** {@inheritDoc} */
    @Override AxisOrGetTsr<V> step( int size );

    /** {@inheritDoc} */
    @Override Tsr<V> get();

    /** {@inheritDoc} */
    @Override Tsr<V> detached();
}
