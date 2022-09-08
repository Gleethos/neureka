package neureka.backend.main.operations.functions;

import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class Identity extends AbstractActivationOperation
{
    public Identity() {
        super(ScalarFun.IDENTITY);
    }
}
