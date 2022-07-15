package neureka.fluent.slicing.states;

public interface ToForTsr<V> extends To<V>
{
    /** {@inheritDoc} */
    StepsOrAxisOrGetTsr<V> to(int index );
}
