package neureka.fluent.building.states;

import neureka.Tensor;
import neureka.common.utility.LogUtil;

import java.util.ArrayList;
import java.util.List;

public interface WithShapeOrScalarOrVectorTensor<V> extends WithShapeOrScalarOrVector<V>
{
    /** {@inheritDoc} */
    @Override
    IterByOrIterFromOrAllTensor<V> withShape(int... shape );

    /** {@inheritDoc} */
    @Override default <N extends Number> IterByOrIterFromOrAllTensor<V> withShape(List<N> shape ) {
        LogUtil.nullArgCheck(shape, "shape", List.class, "Cannot create a tensor without shape!");
        return this.withShape(
                shape.stream().mapToInt(Number::intValue).toArray()
        );
    }

    /** {@inheritDoc} */
    @Override
    Tensor<V> vector(V... values );

    /** {@inheritDoc} */
    @Override default Tensor<V> vector(List<V> values ) {
        return vector( values.toArray( (V[]) new Object[values.size()] ) );
    }

    /** {@inheritDoc} */
    @Override default Tensor<V> vector(Iterable<V> values ) {
        List<V> list = new ArrayList<>();
        values.forEach( list::add );
        return vector( list );
    }

    /** {@inheritDoc} */
    @Override
    Tensor<V> scalar(V value );

}
