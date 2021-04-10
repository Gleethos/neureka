package neureka.utility.slicing.states;



public interface StepsOrThen<V> extends Then<V>
{
    Then<V> step(int size );
}
