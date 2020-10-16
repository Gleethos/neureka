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

    <T> Tsr<T> forward(GraphNode<T> t, Tsr<T> error);

    <T> Tsr<T> backward(GraphNode<T> t, Tsr<T> error);

    Tsr<?> derivative();

    boolean hasForward();

    boolean hasBackward();

    String toString();

}
