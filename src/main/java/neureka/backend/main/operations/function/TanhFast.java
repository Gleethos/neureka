package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.scalar.ScalarFun;

public class TanhFast extends AbstractActivationOperation
{
    public TanhFast() {
        super(ScalarFun.TANH_FAST);
    }
}
