package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.functions.CPUBiFun;

public class CPUBiElementWisePower extends CPUBiElementWise
{
    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return Math.pow( a, b ); }
            @Override public float invoke(float a, float b) { return (float) Math.pow( a, b ); }
            @Override public int invoke(int a, int b) { return (int) Math.round(Math.pow( a, b )); }
            @Override public long invoke(long a, long b) { return Math.round(Math.pow( a, b )); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return b * Math.pow( a, b - 1 ); }
            @Override public float invoke(float a, float b) { return (float) (b * Math.pow( a, b - 1 )); }
            @Override public int invoke(int a, int b) { return (int) Math.round(b * Math.pow( a, b - 1 )); }
            @Override public long invoke(long a, long b) { return Math.round(b * Math.pow( a, b - 1 )); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return Math.pow( a, b ) * Math.log( a ); }
            @Override public float invoke(float a, float b) { return (float) (Math.pow( a, b ) * Math.log( a )); }
            @Override public int invoke(int a, int b) { return (int) Math.round(Math.pow( a, b ) * Math.log( a )); }
            @Override public long invoke(long a, long b) { return Math.round(Math.pow( a, b ) * Math.log( a )); }
        };
    }
}
