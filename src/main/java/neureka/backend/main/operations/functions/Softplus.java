package neureka.backend.main.operations.functions;

import neureka.backend.main.implementations.fun.api.ScalarFun;

/**
 *  SoftPlus is a smooth approximation to the ReLU function and can be used
 *  to constrain the output of a machine to always be positive.
 */
public final class Softplus extends AbstractActivationOperation
{
    public Softplus() {
        super(ScalarFun.SOFTPLUS);
    }
}
