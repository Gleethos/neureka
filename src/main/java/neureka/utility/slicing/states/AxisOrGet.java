package neureka.utility.slicing.states;


import neureka.Tsr;
import neureka.utility.slicing.SliceBuilder;

/**
 *  This is the starting point of the call transition graph exposed by the slice builder API.
 *  It simply defines those method signatures which ought to be called first when using the API.
 *  This interface defines 2 transition paths, namely a route to the end of the call state graph which
 *  triggers the slicing and returns the resulting {@link Tsr} instance... or a call to
 *  the {@link FromOrAt} interface which is the starting point for slicing individual axis of a tensor...
 *
 * @param <V> The type parameter for items of the {@link Tsr} which ought to be sliced.
 */
public interface AxisOrGet<V>  {

    /**
     *  Slicing a tensor ultimately means slicing one or more of its axes!
     *  This method allows one to specify which axis should be sliced next.
     *
     * @param axis The axis which ought to be sliced next.
     * @return The fluent axis slicing API.
     */
    FromOrAt<V> axis( int axis );

    /**
     *  This method concludes the slicing API by performing the actual slicing and
     *  returning the resulting {@link Tsr} instance based on the previously
     *  specified slice configuration...
     *
     * @return An new {@link Tsr} instance which is a slice of the original tensor.
     */
    Tsr<V> get();

}
