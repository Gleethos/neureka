package neureka.utility.fluent.states;

import neureka.Tsr;

public interface IterByOrIterFromOrAll<V>
{

    Tsr<V> iterativelyFilledBy( V... values );

    To<V> iterativelyFilledFrom( V index );

    Tsr<V> all( V value );

}
