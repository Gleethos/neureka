package neureka.backend.main.operations.functions;

import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class Tanh extends AbstractActivationOperation
{
    public Tanh() {
        super(ScalarFun.TANH);
    }
}

