package neureka.backend.main.operations.functions;

import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class Sigmoid extends AbstractActivationOperation
{
    public Sigmoid() {
        super(ScalarFun.SIGMOID);
    }
}
