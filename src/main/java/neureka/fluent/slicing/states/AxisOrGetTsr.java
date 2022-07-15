package neureka.fluent.slicing.states;

import neureka.Tsr;

public interface AxisOrGetTsr<V> extends AxisOrGet<V>
{
    /** {@inheritDoc} */
    @Override FromOrAtTsr<V> axis( int axis );

    /** {@inheritDoc} */
    @Override Tsr<V> get();

    /** {@inheritDoc} */
    @Override Tsr<V> detached();

}
