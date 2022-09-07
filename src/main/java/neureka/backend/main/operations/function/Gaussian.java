package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public final class Gaussian extends AbstractActivationOperation
{
    public Gaussian() {
        super(ActivationFun.GAUSSIAN);
    }
}
