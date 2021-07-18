package neureka.utility.fluent.states;

import neureka.Tsr;
import neureka.ndim.Initializer;

public interface IterByOrIterFromOrAll<V>
{

    Tsr<V> andFill( V... values );

    Tsr<V> andWhere( Initializer<V> initializer );

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
