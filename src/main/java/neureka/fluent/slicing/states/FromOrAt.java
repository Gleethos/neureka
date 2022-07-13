package neureka.fluent.slicing.states;


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
    /**
     *  This is the starting point for defining the slice range of a specified axis within
     *  the method chain/graph exposed by the slice builder API.
     *  I receives the index at which the slice range should start.
     *
     * @param index A valid index in the current axis from which the slice should start.
     * @return The next step in the slicing API which expects one to specify the end of the slice range.
     */
    To<V> from( int index );

    /**
     *  This is a convenience method replacing "{@code from(i).to(i)}", meaning that
     *  it simply slices a single axis from the original tensor at the specified index.
     *
     * @param index The index which ought to be sliced.
     * @return The next step in the slicing API which allows one to slice another axis or simply
     *         perform the actual slicing and get the tensor.
     */
    AxisOrGet<V> at( int index );

    /**
     *  This is a convenience method replacing "{@code from(0).to(axisSize-1)}", meaning that
     *  it simply slices the whole current axis from the original tensor.
     *
     * @return The next step in the slicing API which allows one to slice another axis or simply
     *         perform the actual slicing and get the tensor.
     */
    AxisOrGet<V> all();
}
