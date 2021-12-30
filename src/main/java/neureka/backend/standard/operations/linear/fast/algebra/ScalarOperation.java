/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.algebra;

/**
 */
public interface ScalarOperation {

    interface Addition<T, N extends Comparable<N>> extends ScalarOperation {

        /**
         * @return <code>this + scalarAddend</code>.
         */
        T add(N scalarAddend);

    }

    interface Division<T, N extends Comparable<N>> extends ScalarOperation {

        /**
         * @return <code>this / scalarDivisor</code>.
         */
        T divide(N scalarDivisor);

    }

    interface Multiplication<T, N extends Comparable<N>> extends ScalarOperation {

        /**
         * @return <code>this * multiplicand</code>.
         */
        T multiply(N scalarMultiplicand);

    }

    interface Subtraction<T, N extends Comparable<N>> extends ScalarOperation {

        /**
         * @return <code>this - scalarSubtrahend</code>.
         */
        T subtract(N scalarSubtrahend);

    }

}
