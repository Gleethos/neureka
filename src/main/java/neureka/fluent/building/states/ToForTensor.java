package neureka.fluent.building.states;

public interface ToForTensor<V> extends To<V>
{
    /** {@inheritDoc} */
    @Override
    StepForTensor<V> to( V index );
}
