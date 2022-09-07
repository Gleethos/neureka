package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public final class Absolute extends AbstractActivationOperation
{
    public Absolute() {
        super(ScalarFun.ABSOLUTE);
    }
}
