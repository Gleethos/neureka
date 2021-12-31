/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

/**
 * A (fixed size) 2-dimensional data structure.
 *
 */
public interface Structure2D extends Size {

    static long column(final long index, final long structure) {
        return index / structure;
    }

    static long index(final long structure, final long row, final long column) {
        return row + column * structure;
    }

    static long row(final long index, final long structure) {
        return index % structure;
    }

    /**
     * count() == countRows() * countColumns()
     */
    default int size() {
        return (int) (this.numberOfRows() * this.numberOfColumns());
    }

    /**
     * @return The number of columns
     */
    int numberOfColumns();

    /**
     * @return The number of rows
     */
    int numberOfRows();

    default int getColDim() {
        return Math.toIntExact(this.numberOfColumns());
    }

    default int getRowDim() {
        return Math.toIntExact(this.numberOfRows());
    }

}
