/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.array.DenseArray;
import neureka.backend.standard.operations.linear.fast.array.F64Array;
import neureka.backend.standard.operations.linear.fast.function.FunctionSet;
import neureka.backend.standard.operations.linear.fast.function.PrimitiveFun;

abstract class PrimitiveFactory<I extends Core<Double>> implements MatrixCore.Factory<Double, I> {

    public DenseArray.Factory<Double> array() {
        return F64Array.FACTORY;
    }

    public final FunctionSet<Double> forFunctions() {
        return PrimitiveFun.getSet();
    }

    @Override
    public final I makeFilled(
            final long rows,
            final long columns,
            Object data
    ) {
        return this.make(rows, columns, data);
    }

}
