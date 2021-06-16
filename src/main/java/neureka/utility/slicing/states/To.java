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
    /**
     *  This is the second part for defining the slice range of a specified axis within
     *  the call transition graph exposed by the slice fluent builder API.
     *  This method is the only transition path possible for this interface.
     *  It is leads to the {@link StepsOrAxisOrGet} interface
     *  which, as the name suggests, offers 3 further call transition paths.
     *  This method simply expects the completion of a specified slice range for the current axis.
     *
     * @param index The position where the range should end.
     * @return The next step in the call transition graph of this fluent slice builder API.
     */
    StepsOrAxisOrGet<V> to(int index );
}
