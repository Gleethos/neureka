package neureka.fluent.building.states;

import neureka.Tsr;
import neureka.common.utility.LogUtil;

import java.util.ArrayList;
import java.util.List;

public interface WithShapeOrScalarOrVectorTsr<V> extends WithShapeOrScalarOrVector<V>
{
    /** {@inheritDoc} */
    @Override IterByOrIterFromOrAllTsr<V> withShape( int... shape );

    /** {@inheritDoc} */
    @Override default <N extends Number> IterByOrIterFromOrAllTsr<V> withShape( List<N> shape ) {
        LogUtil.nullArgCheck(shape, "shape", List.class, "Cannot create a tensor without shape!");
        return this.withShape(
                shape.stream().mapToInt(Number::intValue).toArray()
        );
    }

    /** {@inheritDoc} */
    @Override Tsr<V> vector( V... values );

    /** {@inheritDoc} */
    @Override default Tsr<V> vector( List<V> values ) {
        return vector( values.toArray( (V[]) new Object[values.size()] ) );
    }

    /** {@inheritDoc} */
    @Override default Tsr<V> vector( Iterable<V> values ) {
        List<V> list = new ArrayList<>();
        values.forEach( list::add );
        return vector( list );
    }

    /** {@inheritDoc} */
    @Override Tsr<V> scalar( V value );

}
