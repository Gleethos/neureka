package neureka.utility.fluent.states;

import neureka.Tsr;

public interface IterByOrIterFromOrAll<V>
{

    Tsr<V> iterativelyFilledBy( V... values );

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

}
