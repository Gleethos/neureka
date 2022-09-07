package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

/**
 *  The GELU activation function is based on the standard Gaussian cumulative distribution function
 *  and is defined as {@code x Φ( x )} and implemented as {@code x * sigmoid(x * 1.702)}.
 *  The GELU non-linearity weighs inputs by their percentile,
 *  rather than gates inputs by their sign as in ReLUs.
 *  Consequently, the GELU can be thought of as a smoother ReLU.
 */
public class GeLU extends AbstractActivationOperation
{
    public GeLU() {
        super(ScalarFun.GELU);
    }
}
