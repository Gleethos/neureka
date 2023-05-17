package neureka.autograd;

import neureka.Tsr;
import neureka.backend.main.memory.MemUtil;

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
    private int _received;
    private final Tsr<V> _accumulatedError;
    private final int _generation;

    public PendingError( Tsr<V> error, int toBeReceived, int generation ) {
        _expectedToBeReceived = toBeReceived;
        _received = 1; // 1 because the first error value is already given to the constructor.
        _accumulatedError = error;
        _generation = generation;
    }

    public void accumulate( Tsr<?> error ) {
        Tsr[] inputs = { _accumulatedError, error };
        MemUtil.keep( inputs, () -> {
                    _accumulatedError.mut().plusAssign((Tsr<V>)error);
                    return null;
                });
        _received++;
        if ( _received > _expectedToBeReceived ) {
            throw new IllegalStateException(
                    "Received more error values than expected! " +
                    "Expected: " + _expectedToBeReceived + ", " +
                    "Received: " + _received + "."
            );
        }
    }

    public boolean isFullyAccumulated() {
        return _received == _expectedToBeReceived;
    }

    public int getGeneration() { return _generation; }

    public String toString() {
        return this.getClass().getSimpleName()+"[" +
                    "received=" + _received + "," +
                    "accumulatedError=" + _accumulatedError + "," +
                    "generation=" + _generation +
                "]";
    }

    public int getReceived() { return _received; }

    public int getExpectedToBeReceived() { return _expectedToBeReceived; }

    public Tsr<V> getAccumulatedError() { return _accumulatedError; }

}
