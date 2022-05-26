package neureka.autograd;

import neureka.Tsr;

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

    public GraphNode<V> node() { return node; }

    public Tsr<V> error() { return error; }

}
