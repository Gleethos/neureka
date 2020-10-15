package neureka.autograd;

import neureka.Tsr;

public interface ADAgent
{

    <T> Tsr<T> forward(GraphNode t, Tsr<T> error);

    <T> Tsr<T> backward(GraphNode t, Tsr<T> error);

    Tsr<?> derivative();

    boolean isForward();

    boolean hasBackward();

    String toString();


}
