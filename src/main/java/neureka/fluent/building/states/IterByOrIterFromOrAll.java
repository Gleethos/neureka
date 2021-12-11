package neureka.fluent.building.states;

import neureka.Tsr;
import neureka.ndim.Initializer;

public interface IterByOrIterFromOrAll<V>
{
    /**
     *  Provide an array of values which will be used to fill
     *  the {@link Tsr} instance returned by this last fluent builder method.
     *  If the configured tensor is larger than the number of provided
     *  elements, then they will simply be read multiple times until
     *  the result has been sufficiently populated.
     *
     * @param values The values which will be used to populate the {@link Tsr} instance returned by this method.
     * @return The final result of the fluent tensor builder API having a tensor filled with custom values.
     */
    Tsr<V> andFill( V... values );

    /**
     *  Pass a lambda to this method which will be used to populate the {@link Tsr}
     *  built by this fluent builder API based on the indices of the tensor.
     *  The lambda will receive the absolute index ranging from 0 to {@link Tsr#size()}
     *  as well as an array of shape based nd-indices which can be used to
     *  initialize the underlying data of the {@link Tsr} more selectively.
     *
     * @param initializer A data initialization lambda translating the indices to custom values.
     * @return The resulting {@link Tsr} instance populated by the provided {@link Initializer}.
     */
    Tsr<V> andWhere( Initializer<V> initializer );

    /**
     *  This part of the builder API allows for specifying a range which starts from the
     *  provided value and will end at the value specified in the next
     *  builder step returned by this method.
     *  If the number in the created range is not sufficiently large enough to
     *  fully populate the final {@link Tsr} instance built by this API, then the resulting
     *  range will fill the underlying data array of the tensor recurrently.
     *
     * @param index The start of the range which ought to supply the {@link Tsr} instance built by this API.
     * @return The next step in the builder method chain which expects to receive the end point of the range.
     */
    To<V> iterativelyFilledFrom( V index );

    /**
     *  This method creates and return a {@link Tsr} instance which
     *  will be homogeneously filled by the the provided value irrespective
     *  of the previously defined shape.
     *
     * @param value The value which ought to populate the entire {@link Tsr}.
     * @return The homogeneously populated {@link Tsr} instance.
     */
    Tsr<V> all( V value );

    /**
     *  This method creates and return a {@link Tsr} instance which
     *  will be filled based on the provided seed object.
     *  The seed can be any object whose hash will serve as a basis for
     *  supplying the tensor with random data...
     *
     * @param seed The seed based on which the value for populating the entire {@link Tsr} will be generated.
     * @return The pseudo randomly populated {@link Tsr} instance.
     */
    Tsr<V> andSeed( Object seed );

}
