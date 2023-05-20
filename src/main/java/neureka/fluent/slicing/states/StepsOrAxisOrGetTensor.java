package neureka.fluent.slicing.states;

import neureka.Tensor;

public interface StepsOrAxisOrGetTensor<V> extends StepsOrAxisOrGet<V>, AxisOrGetTensor<V>
{
    /** {@inheritDoc} */
    @Override
    AxisOrGetTensor<V> step(int size );

    /** {@inheritDoc} */
    @Override
    Tensor<V> get();

    /** {@inheritDoc} */
    @Override
    Tensor<V> detached();
}
