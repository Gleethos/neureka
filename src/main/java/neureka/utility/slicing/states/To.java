package neureka.utility.slicing.states;

/**
 *  This is the second part for defining the slice range of a specified axis within
 *  the call transition graph exposed by the slice builder API.
 *  This interface defines only 1 transition path, namely a route to the {@link StepsOrAxisOrGet} interface
 *  which, as the name suggests, offers 3 further call transition paths.
 *
 * @param <V> The type parameter for items of the {@link neureka.Tsr} which ought to be sliced.
 */
public interface To<V>
{
    StepsOrAxisOrGet<V> to(int index );
}
