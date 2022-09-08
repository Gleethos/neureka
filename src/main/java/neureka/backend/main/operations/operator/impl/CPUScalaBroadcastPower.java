package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.functions.CPUBiFun;

public class CPUScalaBroadcastPower extends CPUScalarBroadcast
{
    /*
        This ishow it used to be:
        .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> Math.pow( a, b ),
                        ( a, b ) -> b * Math.pow( a, b - 1 ), // Deriving at input 0
                        ( a, b ) -> Math.pow( a, b ) * Math.log( a ) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> (float) Math.pow( a, b ),
                        ( a, b ) -> (float) (b * Math.pow( a, b - 1 )), // Deriving at input 0
                        ( a, b ) -> (float) (Math.pow( a, b ) * Math.log( a )) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> (int) Math.round(Math.pow( a, b )),
                        ( a, b ) -> (int) Math.round(b * Math.pow( a, b - 1 )), // Deriving at input 0
                        ( a, b ) -> (int) Math.round(Math.pow( a, b ) * Math.log( a )) // deriving input 1
                    ))
     */

    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return Math.pow( a, b ); }
            @Override public float   invoke(float a, float b) { return (float) Math.pow( a, b ); }
            @Override public int     invoke(int a, int b) { return (int) Math.round(Math.pow( a, b )); }
            @Override public long    invoke(long a, long b) { return Math.round(Math.pow( a, b )); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return b * Math.pow( a, b - 1 ); }
            @Override public float   invoke(float a, float b) { return (float) (b * Math.pow( a, b - 1 )); }
            @Override public int     invoke(int a, int b) { return (int) Math.round(b * Math.pow( a, b - 1 )); }
            @Override public long    invoke(long a, long b) { return Math.round(b * Math.pow( a, b - 1 )); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return Math.pow( a, b ) * Math.log( a ); }
            @Override public float   invoke(float a, float b) { return (float) (Math.pow( a, b ) * Math.log( a )); }
            @Override public int     invoke(int a, int b) { return (int) Math.round(Math.pow( a, b ) * Math.log( a )); }
            @Override public long    invoke(long a, long b) { return Math.round(Math.pow( a, b ) * Math.log( a )); }
        };
    }
}
