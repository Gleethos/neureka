package neureka.utility.slicing.states;



public interface FromOrAt<V>
{
    To<V> from(int index );
    AxisOrGet<V> at(int index );
}
