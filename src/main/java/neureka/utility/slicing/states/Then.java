package neureka.utility.slicing.states;


import neureka.Tsr;
import neureka.utility.slicing.SliceBuilder;

public interface Then<V>  {

    SliceBuilder<V> then();

    Tsr<V> get();

}
