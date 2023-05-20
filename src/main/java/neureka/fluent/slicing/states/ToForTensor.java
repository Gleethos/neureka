package neureka.fluent.slicing.states;

public interface ToForTensor<V> extends To<V>
{
    /** {@inheritDoc} */
    StepsOrAxisOrGetTensor<V> to( int index );
}
