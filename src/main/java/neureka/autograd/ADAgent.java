package neureka.autograd;

import neureka.Tsr;

/**
 *  ADAgent stands for "Auto-Differentiation-Agent", meaning
 *  that implementations of this class are responsible for managing
 *  forward- and reverse- mode differentiation actions.
 *  These actions are accessible through the "forward(...)"
 *  and "backward(...)" method which are being triggered
 *  by instances of the GraphNode class during propagation.
 */
public interface ADAgent
{

    /**
     *  The auto-differentiation forward pass of an ADAgent
     *  propagate partial differentiations forward into the computation graph.
     *
     * @param target The node which is targeted to hold the partial derivative
     * @param derivative
     * @param <T>
     * @return
     */
    <T> Tsr<T> forward( GraphNode<T> target, Tsr<T> derivative );

    <T> Tsr<T> backward( GraphNode<T> target, Tsr<T> error );

    Tsr<?> derivative();

    boolean hasForward();

    boolean hasBackward();

    String toString();

}
