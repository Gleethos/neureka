package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public final class ReLU extends AbstractActivationOperation
{
    public ReLU() {
        super(ScalarFun.RELU);
    }
}
