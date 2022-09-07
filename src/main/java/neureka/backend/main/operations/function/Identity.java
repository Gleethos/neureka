package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public final class Identity extends AbstractActivationOperation
{
    public Identity() {
        super(ScalarFun.IDENTITY);
    }
}
