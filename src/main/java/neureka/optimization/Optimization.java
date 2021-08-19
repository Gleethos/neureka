package neureka.optimization;

import neureka.Tsr;

public interface Optimization<V> {

    Tsr<V> optimize( Tsr<V> w );

}
