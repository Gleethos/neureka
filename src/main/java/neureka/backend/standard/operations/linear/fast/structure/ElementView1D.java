/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

import neureka.backend.standard.operations.linear.fast.ProgrammingError;

import java.util.Comparator;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public interface ElementView1D<N extends Comparable<N>, V extends ElementView1D<N, V>>
        extends AccessScalar<N>, Iterable<V>, Iterator<V>, Spliterator<V>, Comparable<V> {

    int CHARACTERISTICS = Spliterator.CONCURRENT | Spliterator.DISTINCT | Spliterator.IMMUTABLE | Spliterator.NONNULL | Spliterator.ORDERED | Spliterator.SIZED
            | Spliterator.SORTED | Spliterator.SUBSIZED;

    default int characteristics() {
        return CHARACTERISTICS;
    }

    default int compareTo(final V other) {
        return Long.compare(this.index(), other.index());
    }

    default void forEachRemaining(final Consumer<? super V> action) {
        Spliterator.super.forEachRemaining(action);
    }

    default Comparator<? super V> getComparator() {
        return null;
    }

    boolean hasPrevious();

    long index();

    V iterator();

    default long nextIndex() {
        return this.index() + 1L;
    }

    default long previousIndex() {
        return this.index() - 1L;
    }

    default void remove() {
        ProgrammingError.throwForUnsupportedOptionalOperation();
    }

    default boolean step() {
        if (this.hasNext()) {
            this.next();
            return true;
        } else {
            return false;
        }
    }

    default Stream<V> stream() {
        return StreamSupport.stream(this, false);
    }

    /**
     * @deprecated v48 Use {@link #stream()} instead
     */
    @Deprecated
    default Stream<V> stream(final boolean parallel) {
        return this.stream();
    }

    default boolean tryAdvance(final Consumer<? super V> action) {
        if (this.hasNext()) {
            action.accept(this.next());
            return true;
        } else {
            return false;
        }
    }

    V trySplit();

}
