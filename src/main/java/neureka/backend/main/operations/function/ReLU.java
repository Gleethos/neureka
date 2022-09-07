package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public final class ReLU extends AbstractActivationOperation
{
    public ReLU() {
        super(ActivationFun.RELU);
    }
}
