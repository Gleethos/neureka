package neureka.autograd;

import neureka.Tsr;

/**
 *  This is simply a wrapper for useful information needed by implementations of
 *  the {@link ADAction} and {@link ADAgent} interfaces to perform error propagation.
 *
 * @param <V>
 */
public class Target<V>
{
    private final int index;
    private final GraphNode<V> node;
    private final Tsr<V> error;

    Target(int index, GraphNode<V> node, Tsr<V> error) {
        this.index = index;
        this.node = node;
        this.error = error;
    }

    public int index() { return index; }

    /**
     * @return The targeted graph node of the tensor towards the error should be propagated.
     */
    public GraphNode<V> node() { return node; }

    public Tsr<V> error() { return error; }

}
