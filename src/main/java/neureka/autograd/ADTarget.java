package neureka.autograd;

import neureka.Tensor;

/**
 *  This is simply a wrapper for useful information needed by implementations of
 *  the {@link ADAction} and {@link ADAction} interfaces to perform error propagation.
 *  The class exposes the targeted index and graph node of the input towards
 *  a provided error should be propagated.
 *
 * @param <V> The data type of the tensor.
 */
public final class ADTarget<V>
{
    private final int _inputIndex;
    private final GraphNode<V> _node;
    private final Tensor<V> _error;

    ADTarget( int inputIndex, GraphNode<V> node, Tensor<V> error ) {
        _inputIndex = inputIndex;
        _node = node;
        _error = error;
    }

    /**
     * @return The index of the input targeted for propagation.
     */
    public int inputIndex() { return _inputIndex; }

    /**
     * @return The targeted graph node of the tensor towards the error should be propagated.
     */
    public GraphNode<V> node() { return _node; }

    public Tensor<V> error() { return _error; }

}
