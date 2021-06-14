package neureka.utility.fluent.states;

import neureka.Tsr;

public interface WithShapeOrScalarOrVector<V>
{

    IterByOrIterFromOrAll<V> withShape( int... shape );

    Tsr<V> vector( V... values );

    Tsr<V> scalar( V value );

}
