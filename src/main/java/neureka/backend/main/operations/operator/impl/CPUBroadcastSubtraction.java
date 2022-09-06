package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.algorithms.internal.Fun;

public class CPUBroadcastSubtraction extends CPUBroadcast
{
    public CPUBroadcastSubtraction() {
        super(
            CPUBroadcast.implementationForCPU()
            .with(Fun.F64F64ToF64.triple(
                ( a, b ) -> a - b,
                // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                ( a, b ) -> a + b, // Deriving at input 0
                ( a, b ) -> a - b // deriving input 1
            ))
            .with(Fun.F32F32ToF32.triple(
                ( a, b ) -> a - b,
                // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                ( a, b ) -> a + b, // Deriving at input 0
                ( a, b ) -> a - b // deriving input 1
            ))
        );
    }
}
