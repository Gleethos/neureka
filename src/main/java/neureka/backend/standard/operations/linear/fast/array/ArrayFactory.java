/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.function.FunctionSet;
import neureka.backend.standard.operations.linear.fast.structure.FactorySupplement;

abstract class ArrayFactory<N extends Comparable<N>, I extends BasicArray<N>>
        implements FactorySupplement
{

    public abstract FunctionSet<N> forFunctions();
 
}
