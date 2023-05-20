package neureka.fluent.slicing.states;

import neureka.Tensor;

public interface AxisOrGetTensor<V> extends AxisOrGet<V>
{
    /** {@inheritDoc} */
    @Override
    FromOrAtTensor<V> axis(int axis );

    /** {@inheritDoc} */
    @Override
    Tensor<V> get();

    /** {@inheritDoc} */
    @Override
    Tensor<V> detached();

}
