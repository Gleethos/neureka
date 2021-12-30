/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.function.FunctionSet;
import neureka.backend.standard.operations.linear.fast.matrix.Matrix2D;
import neureka.backend.standard.operations.linear.fast.structure.*;
import neureka.backend.standard.operations.linear.fast.type.NumberDefinition;

public interface MatrixCore<N extends Comparable<N>>
        extends
            Matrix2D<N, MatrixCore<N>>,
            Access2D.Collectable<N, TransformableRegion<N>>,
            Structure2D,
            Structure1D
{
    interface Factory<N extends Comparable<N>, I extends Core<N>> extends Factory2D.Dense<I> {

        FunctionSet<N> forFunctions();

    }

    static int firstInColumn(final Access1D<?> matrix, final int col, final int defaultAndMinimum) {
        return matrix instanceof MatrixCore<?> ? Math.max(((MatrixCore<?>) matrix).firstInColumn(col), defaultAndMinimum) : defaultAndMinimum;
    }

    static int firstInRow(final Access1D<?> matrix, final int row, final int defaultAndMinimum) {
        return matrix instanceof MatrixCore<?> ? Math.max(((MatrixCore<?>) matrix).firstInRow(row), defaultAndMinimum) : defaultAndMinimum;
    }

    static int limitOfColumn(final Access1D<?> matrix, final int col, final int defaultAndMaximum) {
        return matrix instanceof MatrixCore<?> ? Math.min(((MatrixCore<?>) matrix).limitOfColumn(col), defaultAndMaximum) : defaultAndMaximum;
    }

    static int limitOfRow(final Access1D<?> matrix, final int row, final int defaultAndMaximum) {
        return matrix instanceof MatrixCore<?> ? Math.min(((MatrixCore<?>) matrix).limitOfRow(row), defaultAndMaximum) : defaultAndMaximum;
    }

    default MatrixCore<N> add(final N scalarAddend) {
        return new UnaryOperationCore<>(
                            this,
                            this.factory().forFunctions().multiplication().second(scalarAddend)
                    );
    }

    default MatrixCore<N> divide(final N scalarDivisor) {
        return new UnaryOperationCore<>(
                this,
                this.factory().forFunctions().multiplication().second(scalarDivisor)
        );
    }

    default double doubleValue(final long row, final long col) {
        return NumberDefinition.doubleValue(this.getAt(row, col));
    }

    /**
     * The default value is simply <code>0</code>, and if all elements are zeros then
     * <code>this.countRows()</code>.
     *
     * @param col The column index
     * @return The row index of the first non-zero element in the specified column
     */
    default int firstInColumn(final int col) {
        return 0;
    }

    /**
     * The default value is simply <code>0</code>, and if all elements are zeros then
     * <code>this.countColumns()</code>.
     *
     * @return The column index of the first non-zero element in the specified row
     */
    default int firstInRow(final int row) {
        return 0;
    }

    /**
     * The default value is simply <code>this.countRows()</code>, and if all elements are zeros then
     * <code>0</code>.
     *
     * @return The row index of the first zero element, after all non-zeros, in the specified column (index of
     *         the last non-zero + 1)
     */
    default int limitOfColumn(final int col) {
        return (int) this.numberOfRows();
    }

    /**
     * The default value is simply <code>this.countColumns()</code>, and if all elements are zeros then
     * <code>0</code>.
     *
     * @return The column index of the first zero element, after all non-zeros, in the specified row (index of
     *         the last non-zero + 1)
     */
    default int limitOfRow(final int row) {
        return (int) this.numberOfColumns();
    }

    default void multiplyAccess(final Access1D<N> right, final TransformableRegion<N> target) {
        target.fillByMultiplying(this, right);
    }

    default MatrixCore<N> multiply(final MatrixCore<N> right, Object otherData) {

        long tmpCountRows = this.numberOfRows();
        long tmpCountColumns = right.numberOfColumns();

        Core<N> retVal = this.factory().make(tmpCountRows, tmpCountColumns, otherData);

        this.multiplyAccess(right, retVal);

        return retVal;
    }

    default MatrixCore<N> multiply(final N scalarMultiplicand) {
        return new UnaryOperationCore<>(
                this,
                this.factory().forFunctions().multiplication().second(scalarMultiplicand)
        );
    }

    Factory<N, ?> factory();

    default MatrixCore<N> subtract(final MatrixCore<N> subtrahend) {
        throw new IllegalStateException();
        //return this.onMatching(this.factory().forFunctions().subtraction(), subtrahend).collect(this.factory());
    }

    default MatrixCore<N> subtract(final N scalarSubtrahend) {
        throw new IllegalStateException();
    }

    default void supplyToTrans(final TransformableRegion<N> receiver) {
        receiver.fillMatchingAccess(this);
    }

}
