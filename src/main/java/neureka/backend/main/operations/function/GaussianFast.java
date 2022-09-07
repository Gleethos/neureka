package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public class GaussianFast extends AbstractActivationOperation
{
    public GaussianFast() {
        super(ActivationFun.GAUSSIAN_FAST);
    }
}
