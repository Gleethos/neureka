package neureka.fluent.building.states;

import neureka.Tensor;
import neureka.ndim.Filler;

import java.util.List;

public interface IterByOrIterFromOrAllTensor<V> extends IterByOrIterFromOrAll<V>
{
    /** {@inheritDoc} */
    @Override
    Tensor<V> andFill(V... values );

    /** {@inheritDoc} */
    @Override default Tensor<V> andFill(List<V> values ) {
        return this.andFill((V[])values.toArray());
    }

    /** {@inheritDoc} */
    @Override
    Tensor<V> andWhere(Filler<V> filler );

    /** {@inheritDoc} */
    @Override
    ToForTensor<V> andFillFrom(V index );

    /** {@inheritDoc} */
    @Override
    Tensor<V> all(V value );

    /** {@inheritDoc} */
    @Override
    Tensor<V> andSeed(Object seed );

}
