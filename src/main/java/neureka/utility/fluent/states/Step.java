package neureka.utility.fluent.states;

import neureka.Tsr;

public interface Step<V>
{

    Tsr<V> step( double size );

}
