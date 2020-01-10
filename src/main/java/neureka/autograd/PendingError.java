package neureka.autograd;

import neureka.Tsr;
import neureka.calculus.factory.assembly.FunctionBuilder;

public class PendingError {

    private int _toBeReceived;
    private Tsr _error;

    public PendingError(Tsr error, int toBeReceived){
        _toBeReceived = toBeReceived;
        _error = error;
    }

    public void accumulate(Tsr error){
        FunctionBuilder.build(
                "I[0]<-(I[0]+I[1])", false
        ).activate(new Tsr[]{_error, error});
        _toBeReceived--;
    }

    public boolean isFullyAccumulated(){
        return _toBeReceived==0;
    }

    public Tsr getAccumulatedError(){
        return _error;
    }

}
