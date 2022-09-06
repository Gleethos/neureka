package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.algorithms.internal.Fun;

public class CPUBroadcastModulo extends CPUBroadcast
{
    public CPUBroadcastModulo() {
        super(
            CPUBroadcast.implementationForCPU()
            .with(Fun.F64F64ToF64.triple(
                ( a, b ) -> a % b,
                ( a, b ) -> 1 / b, // Deriving at input 0
                ( a, b ) -> -(a / Math.pow(b, 2)) // deriving input 1
            ))
            .with(Fun.F32F32ToF32.triple(
                ( a, b ) -> a % b,
                ( a, b ) -> 1 / b, // Deriving at input 0
                ( a, b ) -> (float) -(a / Math.pow(b, 2)) // deriving input 1
            ))
        );
    }
}
