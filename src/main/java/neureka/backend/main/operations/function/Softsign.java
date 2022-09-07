package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

/**
 *  The softsign function, defined as {@code x / ( 1 + Math.abs( x ) )},
 *  is a computationally cheap 0 centered activation function
 *  which rescales the inputs between -1 and 1, very much like the {@link Tanh} function.
 *  The softsign function converges polynomially and is computationally cheaper than the
 *  tanh function which converges exponentially.
 *  This makes this function a computationally cheap non-exponential quasi {@link Tanh}!
 */
public class Softsign extends AbstractActivationOperation
{
    public Softsign() {
        super(ScalarFun.SOFTSIGN);
    }
}
