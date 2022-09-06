package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.algorithms.internal.Fun;

public class CPUBroadcastAddition extends CPUBroadcast
{
    public CPUBroadcastAddition() {
        super(
            CPUBroadcast.implementationForCPU()
            .with(Fun.F64F64ToF64.triple(
                ( a, b ) -> a + b,
                ( a, b ) -> a + b, // Deriving at input 0
                ( a, b ) -> a + b  // deriving input 1
            ))
            .with(Fun.F32F32ToF32.triple(
                ( a, b ) -> a + b,
                ( a, b ) -> a + b, // Deriving at input 0
                ( a, b ) -> a + b  // deriving input 1
            ))
            .with(Fun.BoolBoolToBool.triple(
                ( a, b ) -> a && b,
                ( a, b ) -> a && b,
                ( a, b ) -> a && b
            ))
            .with(Fun.CharCharToChar.triple(
                ( a, b ) -> (char) (((int)a)+((int)b)),
                ( a, b ) -> (char) (((int)a)+((int)b)),
                ( a, b ) -> (char) (((int)a)+((int)b))
            ))
        );
    }
}
