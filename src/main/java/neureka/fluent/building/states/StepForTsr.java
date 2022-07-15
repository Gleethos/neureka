package neureka.fluent.building.states;

import neureka.Tsr;

public interface StepForTsr<V> extends Step<V>
{
    /** {@inheritDoc} */
    @Override Tsr<V> step(double size );
}
