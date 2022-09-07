package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

public final class Tanh extends AbstractActivationOperation
{
    public Tanh() {
        super(ScalarFun.TANH);
    }
}

