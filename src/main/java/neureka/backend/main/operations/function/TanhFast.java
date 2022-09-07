package neureka.backend.main.operations.function;

import neureka.backend.main.functions.ScalarFun;

public class TanhFast extends AbstractActivationOperation
{
    public TanhFast() {
        super(ScalarFun.TANH_FAST);
    }
}
