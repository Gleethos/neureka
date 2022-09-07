package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

public final class ReLU extends AbstractActivationOperation
{
    public ReLU() {
        super(ScalarFun.RELU);
    }
}
