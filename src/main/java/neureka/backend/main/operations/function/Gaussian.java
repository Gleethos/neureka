package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

public final class Gaussian extends AbstractActivationOperation
{
    public Gaussian() {
        super(ScalarFun.GAUSSIAN);
    }
}
