package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public final class Sigmoid extends AbstractActivationOperation
{
    public Sigmoid() {
        super(ScalarFun.SIGMOID);
    }
}
