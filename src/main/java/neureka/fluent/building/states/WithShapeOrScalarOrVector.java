package neureka.fluent.building.states;

import neureka.Nda;
import neureka.Tensor;

import java.util.List;

public interface WithShapeOrScalarOrVector<V>
{
    /**
     *  Define a tensor shape by passing an array of int values to this method,
     *  which represent the shape of the {@link Tensor} that should be built.
     *  This should be called immediately after having specified the type of the tensor.
     *
     * @param shape The shape array of the {@link Tensor} that should be built.
     * @return The next step in the call transition graph of this fluent builder API.
     */
    IterByOrIterFromOrAll<V> withShape( int... shape );

    /**
     *  Define a tensor shape by passing a list of numbers to this method,
     *  which represent the shape of the {@link Tensor} that should be built.
     *  This should be called immediately after having specified the type of the tensor.
     *
     * @param shape The shape list of the {@link Tensor} that should be built.
     * @return The next step in the call transition graph of this fluent builder API.
     */
    <N extends Number> IterByOrIterFromOrAll<V> withShape( List<N> shape );


    /**
     *  This method creates and returns a vector {@link Tensor} instance
     *  which wraps the provided values.
     *
     * @param values The values which ought to be wrapped by a new vector {@link Tensor} instance.
     * @return A vector {@link Tensor} instance wrapping the provided values.
     */
    Nda<V> vector( V... values );

    /**
     *  This method creates and returns a vector {@link Tensor} instance
     *  which wraps the provided values.
     *
     * @param values The list of values which ought to be turned into a vector.
     * @return A vector representing the provided values.
     */
    Nda<V> vector( List<V> values );

    /**
     *  This method creates and returns a vector {@link Tensor} instance
     *  which wraps the provided values.
     *
     * @param values The list of values which ought to be turned into a vector.
     * @return A vector representing the provided values.
     */
    Nda<V> vector( Iterable<V> values );

    /**
     *  This method created and return a scalar {@link Tensor} instance
     *  which wraps the provided value.
     *
     * @param value The value which ought to be wrapped by a new scalar {@link Tensor} instance.
     * @return A scala {@link Tensor} instance wrapping the provided value.
     */
    Nda<V> scalar( V value );

}
