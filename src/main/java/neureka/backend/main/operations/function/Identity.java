package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

public final class Identity extends AbstractActivationOperation
{
    public Identity() {
        super(ScalarFun.IDENTITY);
    }
}
