package neureka.autograd;

import neureka.Tsr;

/**
 *  This interface is the declaration for
 *  lambda actions for both the {@link #act(ADTarget)} method of the {@link ADAgent} interface. <br><br>
 *  Implementations of this perform auto-differentiation forwards or backwards along the computation graph.
 *
 *
 * Note: Do not access the {@link GraphNode#getPayload()} of the {@link GraphNode}
 *       passed to implementation of this.
 *       The payload is weakly referenced, meaning that this method can return null!
 */
public interface ADAction
{
    /**
     * @param target A container for the {@link GraphNode} at which the differentiation ought to be performed and error which ought to be used for the forward or backward differentiation.
     * @return The result of a forward or backward mode auto differentiation.
     */
    Tsr<?> act( ADTarget<?> target );
}
