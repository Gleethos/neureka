package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public final class Quadratic extends AbstractActivationOperation
{
    public Quadratic() {
        super(ActivationFun.QUADRATIC);
    }
}
