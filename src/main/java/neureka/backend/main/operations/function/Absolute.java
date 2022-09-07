package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

public final class Absolute extends AbstractActivationOperation
{
    public Absolute() {
        super(ScalarFun.ABSOLUTE);
    }
}
