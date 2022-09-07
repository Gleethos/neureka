package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public final class Absolute extends AbstractActivationOperation
{
    public Absolute() {
        super(ActivationFun.ABSOLUTE);
    }
}
