package neureka.autograd;

import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.assembly.FunctionBuilder;

public final class PendingError<ValType>
{
    private int _toBeReceived;
    private final Tsr<ValType> _accumulatedError;

    public PendingError( Tsr<ValType> error, int toBeReceived ) {
        _toBeReceived = toBeReceived;
        _accumulatedError = error;
    }

    public void accumulate( Tsr<?> error ) {
        new FunctionBuilder(Neureka.get().context()).build(
                "I[ 0 ]<-(I[ 0 ]+I[ 1 ])", false
        ).call( new Tsr[]{ _accumulatedError, error } );
        _toBeReceived--;
    }

    public boolean isFullyAccumulated() {
        return _toBeReceived == 0;
    }

    public String toString() {
        return "PendingError(_toBeReceived=" + this._toBeReceived + ", _accumulatedError=" + this._accumulatedError + ")";
    }

    public int getToBeReceived() {
        return this._toBeReceived;
    }

    public Tsr<ValType> getAccumulatedError() {
        return this._accumulatedError;
    }
}
