package neureka.backend.main.operations.functions;

import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class Gaussian extends AbstractActivationOperation
{
    public Gaussian() {
        super(ScalarFun.GAUSSIAN);
    }
}
