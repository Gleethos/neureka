package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

/**
 *  The Self Gated {@link Softsign} Unit is based on the {@link Softsign} function
 *  (a computationally cheap non-exponential quasi {@link Tanh})
 *  making it a polynomially based version of the {@link GaTU} function which
 *  is itself based on the {@link Tanh} function.
 *  Similar as the {@link Softsign} and {@link Tanh} function {@link GaSU}
 *  is 0 centered and capped by -1 and +1.
 */
public class GaSU extends AbstractActivationOperation
{
    public GaSU() {
        super(ScalarFun.GASU);
    }
}

