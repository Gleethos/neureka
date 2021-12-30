/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.algebra;

/**
 * @see <a href="https://en.wikipedia.org/wiki/Operation_(mathematics)">Operation (mathematics)</a>
 */
public interface Operation {

    /**
     * @see <a href="https://en.wikipedia.org/wiki/Addition">Addition</a>
     */
    interface Addition<T> extends Operation {

        /**
         * @param addend What to add
         * @return <code>this + addend</code>
         */
        T addMat(T addend);

    }

    /**
     * @see <a href="https://en.wikipedia.org/wiki/Division_(mathematics)">Division (mathematics)</a>
     */
    interface Division<T> extends Operation {

        /**
         * @param divisor The divisor
         * @return <code>this / divisor</code>.
         */
        T divide(T divisor);

    }

    /**
     * @see <a href="https://en.wikipedia.org/wiki/Multiplication">Multiplication</a>
     */
    interface Multiplication<T> extends Operation {

        /**
         * @param multiplicand The multiplicand
         * @return <code>this * multiplicand</code>.
         */
        T multiply(T multiplicand, Object otherData);

    }

    /**
     * @see <a href="https://en.wikipedia.org/wiki/Subtraction">Subtraction</a>
     */
    interface Subtraction<T> extends Operation {

        /**
         * @param subtrahend The subtrahend
         * @return <code>this - subtrahend</code>.
         */
        T subtract(T subtrahend);
    }

}
