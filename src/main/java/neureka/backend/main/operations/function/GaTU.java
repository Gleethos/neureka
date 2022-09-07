package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

/**
 *  The Self Gated {@link Tanh} Unit is based on the {@link Tanh}
 *  making it an exponentiation based version of the {@link GaSU} function which
 *  is itself based on the {@link Softsign} function
 *  (a computationally cheap non-exponential quasi {@link Tanh}).
 *  Similar a the {@link Softsign} and {@link Tanh} function {@link GaTU}
 *  is 0 centered and caped by -1 and +1.
 */
public class GaTU  extends AbstractActivationOperation
{
    public GaTU() {
        super(ScalarFun.GATU);
    }
}
