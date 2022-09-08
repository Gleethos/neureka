package neureka.backend.main.operations.functions;

import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class ReLU extends AbstractActivationOperation
{
    public ReLU() {
        super(ScalarFun.RELU);
    }
}
