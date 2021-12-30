/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast;

import neureka.backend.standard.operations.linear.fast.structure.Access2D;
import neureka.backend.standard.operations.linear.fast.structure.Structure1D;
import neureka.backend.standard.operations.linear.fast.structure.Structure2D;

/**
 * Incorrect use of the API. The code needs to be changed. Typically, execution can't continue. Is never
 * declared to be thrown, and should not be caught.
 *
 */
public class ProgrammingError extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public static void throwForMultiplicationNotPossible() {
        throw new ProgrammingError("The column dimension of the left matrix does not match the row dimension of the right matrix!");
    }

    public static void throwForUnsupportedOptionalOperation() {
        throw new UnsupportedOperationException();
    }

    public static void throwIfMultiplicationNotPossible(final Access2D<?> left, final Access2D<?> right) {
        if (left.numberOfColumns() != right.numberOfRows()) {
            ProgrammingError.throwForMultiplicationNotPossible();
        }
    }

    public static void throwIfNotEqualColumnDimensions(final Access2D<?> mtrx1, final Access2D<?> mtrx2) {
        if (mtrx1.numberOfColumns() != mtrx2.numberOfColumns()) {
            throw new ProgrammingError("Column dimensions are not equal!");
        }
    }

    public static void throwIfNotEqualDimensions(final Access2D<?> mtrx1, final Access2D<?> mtrx2) {
        ProgrammingError.throwIfNotEqualRowDimensions(mtrx1, mtrx2);
        ProgrammingError.throwIfNotEqualColumnDimensions(mtrx1, mtrx2);
    }

    public static void throwIfNotEqualRowDimensions(final Structure2D mtrx1, final Structure1D mtrx2) {
        if (mtrx2 instanceof Structure2D) {
            if (mtrx1.numberOfRows() != ((Structure2D) mtrx2).numberOfRows()) {
                throw new ProgrammingError("Row dimensions are not equal!");
            }
        } else if (mtrx1.numberOfRows() != mtrx2.size()) {
            throw new ProgrammingError("Row dimensions are not equal!");
        }
    }

    public ProgrammingError(final String message) {
        super(message);
    }

    public ProgrammingError(final Throwable cause) {
        super(cause);
    }

    @Override
    public String toString() {
        final String retVal = this.getClass().getSimpleName();
        final String tmpMessage = this.getLocalizedMessage();
        return tmpMessage != null ? retVal + ": " + tmpMessage : retVal;
    }

}
