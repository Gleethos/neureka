package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public final class Identity extends AbstractActivationOperation
{
    public Identity() {
        super(ActivationFun.IDENTITY);
    }
}
