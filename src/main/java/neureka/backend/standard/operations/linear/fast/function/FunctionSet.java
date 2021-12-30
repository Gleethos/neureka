/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

/**
 *  A set of predefined/standard functions.
 */
public abstract class FunctionSet<N extends Comparable<N>> {

    protected FunctionSet() { super(); }

    public abstract BinaryFunction<N> addition();

    public abstract BinaryFunction<N> division();

    public abstract BinaryFunction<N> multiplication();

    public abstract BinaryFunction<N> subtraction();

}
