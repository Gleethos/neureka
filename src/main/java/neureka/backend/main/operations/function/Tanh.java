package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public final class Tanh extends AbstractActivationOperation
{
    public Tanh() {
        super(ActivationFun.TANH);
    }
}

