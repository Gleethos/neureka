/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.algebra;

/**
 */
public interface ScalarOperation {

    interface Multiplication<T, N extends Comparable<N>> extends ScalarOperation {

        /**
         * @return <code>this * multiplicand</code>.
         */
        T multiply(N scalarMultiplicand);

    }

}
