package neureka.backend.main.operations.functions;

import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class Absolute extends AbstractActivationOperation
{
    public Absolute() {
        super(ScalarFun.ABSOLUTE);
    }
}
