package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.algorithms.internal.Fun;

public class CPUBroadcastMultiplication extends CPUBroadcast
{
    public CPUBroadcastMultiplication() {
        super(
            CPUBroadcast.implementationForCPU()
            .with(Fun.F64F64ToF64.triple(
                ( a, b ) -> a * b,
                ( a, b ) -> b, // Deriving at input 0
                ( a, b ) -> a  // deriving input 1
            ))
            .with(Fun.F32F32ToF32.triple(
                ( a, b ) -> a * b,
                ( a, b ) -> b, // Deriving at input 0
                ( a, b ) -> a  // deriving input 1
            ))
        );
    }
}
