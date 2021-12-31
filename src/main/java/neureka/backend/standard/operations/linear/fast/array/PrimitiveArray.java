/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.structure.Size;

public abstract class PrimitiveArray extends PlainArray<Double> implements Size {

    PrimitiveArray(final Factory<Double> factory, final int size) {
        super(factory, size);
    }

}
