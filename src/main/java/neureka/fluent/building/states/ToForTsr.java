package neureka.fluent.building.states;

public interface ToForTsr<V> extends To<V>
{
    /** {@inheritDoc} */
    @Override StepForTsr<V> to( V index );
}
