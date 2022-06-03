package neureka.backend.main.algorithms.internal;

import neureka.backend.api.Call;
import neureka.calculus.args.Arg;

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

    default T get( Arg.DerivIdx index ) {
        return this.get( index.get() );
    }

    /**
     * @param call The execution call for which a suitable scalar function ought to be selected.
     * @return An appropriate scalar function for thr provided call.
     */
    default T getFor( Call<?> call ) {
        return this.get( call.get( Arg.DerivIdx.class ).get() );
    }

}
