package neureka.utility.slicing.states;



public interface FromOrAt<V>
{
    To<V> from(int index );
    Then<V> at(int index );
}
