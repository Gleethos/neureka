package neureka.utility.slicing.states;


import neureka.Tsr;
import neureka.utility.slicing.SliceBuilder;

public interface AxisOrGet<V>  {

    FromOrAt<V> axis(int axis );

    Tsr<V> get();

}
