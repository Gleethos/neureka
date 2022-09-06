package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.algorithms.internal.Fun;

public class CPUBroadcastPower extends CPUBroadcast
{
    public CPUBroadcastPower() {
        super(
            CPUBroadcast.implementationForCPU()
            .with(Fun.F64F64ToF64.triple(
                ( a, b ) -> Math.pow(a, b),
                // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                ( a, b ) -> a * Math.pow( a, b - 1  ), // Deriving at input 0
                ( a, b ) -> Math.pow( a, b ) * Math.log(a) // deriving input 1
            ))
            .with(Fun.F32F32ToF32.triple(
                ( a, b ) -> (float) Math.pow(a, b),
                // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                ( a, b ) -> (float) (a * Math.pow( a, b - 1  )), // Deriving at input 0
                ( a, b ) -> (float) (Math.pow( a, b ) * Math.log(a)) // deriving input 1
            ))
        );
    }
}
