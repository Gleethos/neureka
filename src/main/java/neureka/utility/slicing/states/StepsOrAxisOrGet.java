package neureka.utility.slicing.states;



public interface StepsOrAxisOrGet<V> extends AxisOrGet<V>
{
    AxisOrGet<V> step(int size );
}
