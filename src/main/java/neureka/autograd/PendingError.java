package neureka.autograd;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.main.memory.MemUtil;
import neureka.math.parsing.FunctionParser;

/**
 *  A wrapper for a tensor which is used to accumulate error values
 *  during the back propagation phase of the autograd algorithm.
 *  This is a library internal class, do not depend on this.
 *  <p>
 *  The {@link PendingError} class also keeps track of how many
 *  more error values need to be accumulated before the error
 *  value is fully accumulated.
 *
 * @param <V> The data type of the tensor which is used to accumulate error values.
 */
final class PendingError<V>
{
    private final int _expectedToBeReceived;
    private int _toBeReceived;
    private final Tsr<V> _accumulatedError;

    public PendingError( Tsr<V> error, int toBeReceived ) {
        _expectedToBeReceived = toBeReceived;
        _toBeReceived = toBeReceived - 1; // -1 because the first error value is already given to the constructor.
        _accumulatedError = error;
    }

    public void accumulate( Tsr<?> error ) {
        Tsr[] inputs = { _accumulatedError, error };
        MemUtil.keep( inputs, () -> {
                    _accumulatedError.mut().plusAssign((Tsr<V>)error);
                    return null;
                });
        _toBeReceived--;
    }

    public boolean isFullyAccumulated() {
        return _toBeReceived == 0;
    }

    public String toString() {
        return this.getClass().getSimpleName()+"[toBeReceived=" + _toBeReceived + ",accumulatedError=" + _accumulatedError + "]";
    }

    public int getToBeReceived() { return _toBeReceived; }

    public int getExpectedToBeReceived() { return _expectedToBeReceived; }

    public Tsr<V> getAccumulatedError() { return _accumulatedError; }

}
