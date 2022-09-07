package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public final class Tanh extends AbstractActivationOperation
{
    public Tanh() {
        super(ScalarFun.TANH);
    }
}

