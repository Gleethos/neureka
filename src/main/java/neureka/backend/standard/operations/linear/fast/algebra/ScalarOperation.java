/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.algebra;

/**
 */
public interface ScalarOperation {

    interface Addition<T, N extends Comparable<N>> extends ScalarOperation {

    }

    interface Division<T, N extends Comparable<N>> extends ScalarOperation {

    }

    interface Multiplication<T, N extends Comparable<N>> extends ScalarOperation {

        /**
         * @return <code>this * multiplicand</code>.
         */
        T multiply(N scalarMultiplicand);

    }

    interface Subtraction<T, N extends Comparable<N>> extends ScalarOperation {

    }

}
