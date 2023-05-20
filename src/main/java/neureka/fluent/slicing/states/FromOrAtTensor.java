package neureka.fluent.slicing.states;

public interface FromOrAtTensor<V> extends FromOrAt<V>
{
    /** {@inheritDoc} */
    @Override
    ToForTensor<V> from(int index );

    /** {@inheritDoc} */
    @Override
    AxisOrGetTensor<V> at( int index );

    /** {@inheritDoc} */
    @Override
    AxisOrGetTensor<V> all();
}
