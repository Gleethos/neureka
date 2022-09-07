package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

/**
 *  The SiLu activation function, also known as the swish function, is defined as {@code x * sigmoid(x)}.
 *  It is a smooth, non-monotonic function that consistently matches
 *  or outperforms ReLU on deep networks,
 *  it is unbounded above and bounded below.
 */
public class SiLU extends AbstractActivationOperation
{
    public SiLU() {
        super(ScalarFun.SILU);
    }
}
