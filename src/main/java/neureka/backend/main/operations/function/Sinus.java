package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public final class Sinus extends AbstractActivationOperation
{
    public Sinus() {
        super(ActivationFun.SINUS);
    }
}
