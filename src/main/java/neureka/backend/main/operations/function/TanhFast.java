package neureka.backend.main.operations.function;

import neureka.backend.main.operations.function.internal.ActivationFun;

public class TanhFast extends AbstractActivationOperation
{
    public TanhFast() {
        super(ActivationFun.TANH_FAST);
    }
}
