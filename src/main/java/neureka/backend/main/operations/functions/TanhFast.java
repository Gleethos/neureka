package neureka.backend.main.operations.functions;

import neureka.backend.main.implementations.fun.api.ScalarFun;

public class TanhFast extends AbstractActivationOperation
{
    public TanhFast() {
        super(ScalarFun.TANH_FAST);
    }
}
