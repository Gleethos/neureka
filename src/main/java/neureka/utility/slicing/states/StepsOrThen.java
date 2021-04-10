package neureka.utility.slicing.states;



public interface StepsOrThen<V> extends Then<V>
{
    Then<V> steps(int index );
}
