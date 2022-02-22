package neureka.ndim;

/**
 *  Implementations of this ought to map the index of a
 *  tensor entry to a value which should be placed at that entry position.
 *
 * @param <T> The type parameter determining the type of the supplied values.
 */
@FunctionalInterface
public interface Filler<T> {

    T init( int i, int[] index );

}

