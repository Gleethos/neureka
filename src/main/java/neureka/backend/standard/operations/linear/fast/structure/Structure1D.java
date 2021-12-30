/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

/**
 * A (fixed size) 1-dimensional data structure.
 *
 */
public interface Structure1D {

    final class IntIndex implements Comparable<IntIndex> {

        public static IntIndex of(final int index) {
            return new IntIndex(index);
        }

        public final int index;

        public IntIndex(final int anIndex) {
            super();
            index = anIndex;
        }

        @SuppressWarnings("unused")
        private IntIndex() {
            this(-1);
        }

        public int compareTo(final IntIndex ref) {
            return Integer.compare(index, ref.index);
        }

    }

    default int size() {
        return 0;
    }

}
