package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public class GaussianFast extends AbstractActivationOperation
{
    public GaussianFast() {
        super(ScalarFun.GAUSSIAN_FAST);
    }
}
