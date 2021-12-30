/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

public interface Factory2D<I> extends FactorySupplement {


    I copy(Access2D<?> source);

    I makeFilled(long rows, long columns, Object data);

    /**
     * Should only be implemented by factories that always (or primarily) produce dense structures.
     *
     * 
     */
    interface Dense<I extends Structure2D> extends Factory2D<I> {

    }

    I make(long rows, long columns, Object data);

    default I make(final Structure2D shape) {
        return this.make(shape.numberOfRows(), shape.numberOfColumns(), null);
    }

}
