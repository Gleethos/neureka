package neureka.fluent.slicing.states;

public interface StepsOrAxisOrGetTsr<V> extends StepsOrAxisOrGet<V>, AxisOrGetTsr<V>
{
    /** {@inheritDoc} */
    @Override AxisOrGetTsr<V> step( int size );
}
