package neureka.backend.standard.algorithms.internal;

/**
 *  Implementations of this are tuples of scalar functions
 *  where the functions at index 1, 2... are derivatives of the
 *  first function stored at position 0.
 *
 * @param <T> The type of scalar function stored by this tuple.
 */
public interface FunTuple<T extends Fun> {

    T get( int derivativeIndex );

    Class<T> getType();

}
