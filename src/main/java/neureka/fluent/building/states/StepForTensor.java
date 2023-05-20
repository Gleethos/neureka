package neureka.fluent.building.states;

import neureka.Tensor;

public interface StepForTensor<V> extends Step<V>
{
    /** {@inheritDoc} */
    @Override
    Tensor<V> step(double size );
}
