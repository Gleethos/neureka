package neureka.utility.slicing.states;


/**
 *  This is the starting point for defining the slice range of a specified axis within
 *  the call transition graph exposed by the slice builder API.
 *  This interface defines 2 transition paths, namely a route to the {@link To} interface
 *  of the call transition graph, which expects a range to be defined by
 *  calling the methods {@link FromOrAt#from} and {@link To#to},
 *  or a call to the "at" method, which is a shortcut for calling {@link FromOrAt#from} and {@link To#to}
 *  with the same arguments.
 *
 * @param <V> The type parameter for items of the {@link neureka.Tsr} which ought to be sliced.
 */
public interface FromOrAt<V>
{
    To<V> from(int index );
    AxisOrGet<V> at(int index );
}
