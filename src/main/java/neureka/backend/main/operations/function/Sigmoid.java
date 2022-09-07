package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public final class Sigmoid extends AbstractActivationOperation
{
    public Sigmoid() {
        super(ActivationFun.SIGMOID);
    }
}
