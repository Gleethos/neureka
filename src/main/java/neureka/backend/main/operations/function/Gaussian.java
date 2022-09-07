package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public final class Gaussian extends AbstractActivationOperation
{
    public Gaussian() {
        super(ScalarFun.GAUSSIAN);
    }
}
