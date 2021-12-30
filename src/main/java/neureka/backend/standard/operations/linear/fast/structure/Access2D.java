/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

/**
 * 2-dimensional accessor methods
 *
 * @see Access1D
 */
public interface Access2D<N extends Comparable<N>> extends Structure2D, Access1D<N> {

    interface Collectable<N extends Comparable<N>, R> extends Structure2D {

        void supplyToTrans(R receiver);

    }

    static String toString(final Access2D<?> matrix) {

        StringBuilder builder = new StringBuilder();

        int numbRows = Math.toIntExact(matrix.numberOfRows());
        int numbCols = Math.toIntExact(matrix.numberOfColumns());

        builder.append(matrix.getClass().getName());
        builder.append(' ').append('<').append(' ').append(numbRows).append(' ').append('x').append(' ').append(numbCols).append(' ').append('>');

        if (numbRows > 0 && numbCols > 0 && numbRows <= 50 && numbCols <= 50 && numbRows * numbCols <= 200) {

            // First element
            builder.append("\n{ { ").append(matrix.getAt(0, 0));

            // Rest of the first row
            for (int j = 1; j < numbCols; j++) {
                builder.append(",\t").append(matrix.getAt(0, j));
            }

            // For each of the remaining rows
            for (int i = 1; i < numbRows; i++) {

                // First column
                builder.append(" },\n{ ").append(matrix.getAt(i, 0));

                // Remaining columns
                for (int j = 1; j < numbCols; j++) {
                    builder.append(",\t").append(matrix.getAt(i, j));
                }
            }

            // Finish
            builder.append(" } }");
        }

        return builder.toString();
    }

    default double doubleValue(final long index) {
        final long structure = this.numberOfRows();
        return this.doubleValue(Structure2D.row(index, structure), Structure2D.column(index, structure));
    }

    /**
     * Extracts one element of this matrix as a double.
     *
     * @param row A row index.
     * @param col A column index.
     * @return One matrix element
     */
    double doubleValue(long row, long col);

    default float floatValue(final long index) {
        final long structure = this.numberOfRows();
        return this.floatValue(Structure2D.row(index, structure), Structure2D.column(index, structure));
    }

    default float floatValue(final long row, final long col) {
        return (float) this.doubleValue(row, col);
    }

    default N get(final long index) {
        final long tmpStructure = this.numberOfRows();
        return this.getAt(Structure2D.row(index, tmpStructure), Structure2D.column(index, tmpStructure));
    }

    N getAt(long row, long col);

}
