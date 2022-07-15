package neureka.fluent.slicing.states;

public interface FromOrAtTsr<V> extends FromOrAt<V>
{
    /** {@inheritDoc} */
    @Override
    ToForTsr<V> from( int index );

    /** {@inheritDoc} */
    @Override AxisOrGetTsr<V> at( int index );

    /** {@inheritDoc} */
    @Override AxisOrGetTsr<V> all();
}
