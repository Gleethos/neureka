/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.structure.Structure1D;

public abstract class PrimitiveArray extends PlainArray<Double> implements Structure1D {

    PrimitiveArray(final Factory<Double> factory, final int size) {
        super(factory, size);
    }

}
