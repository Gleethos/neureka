package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public final class Quadratic extends AbstractActivationOperation
{
    public Quadratic() {
        super(ScalarFun.QUADRATIC);
    }
}
