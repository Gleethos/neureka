package neureka.fluent.building.states;

import neureka.Nda;
import neureka.Tsr;
import neureka.common.utility.LogUtil;

import java.util.List;

public interface WithShapeOrScalarOrVector<V>
{
    /**
     *  Define a tensor shape by passing an array of int values to this method,
     *  which represent the shape of the {@link Tsr} that should be built.
     *  This should be called immediately after having specified the type of the tensor.
     *
     * @param shape The shape array of the {@link Tsr} that should be built.
     * @return The next step in the call transition graph of this fluent builder API.
     */
    IterByOrIterFromOrAll<V> withShape( int... shape );

    /**
     *  Define a tensor shape by passing a list of numbers to this method,
     *  which represent the shape of the {@link Tsr} that should be built.
     *  This should be called immediately after having specified the type of the tensor.
     *
     * @param shape The shape list of the {@link Tsr} that should be built.
     * @return The next step in the call transition graph of this fluent builder API.
     */
    <N extends Number> IterByOrIterFromOrAll<V> withShape( List<N> shape );


    /**
     *  This method created and return a vector {@link Tsr} instance
     *  which wraps the provided values.
     *
     * @param values The values which ought to be wrapped by a new vector {@link Tsr} instance.
     * @return A vector {@link Tsr} instance wrapping the provided values.
     */
    Nda<V> vector( V... values );

    /**
     *  This method created and return a scalar {@link Tsr} instance
     *  which wraps the provided value.
     *
     * @param value The value which ought to be wrapped by a new scalar {@link Tsr} instance.
     * @return A scala {@link Tsr} instance wrapping the provided value.
     */
    Nda<V> scalar( V value );

}
