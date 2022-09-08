package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.functions.CPUBiFun;

public class CPUBroadcastPower extends CPUBroadcast
{
    public CPUBroadcastPower() {}

    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return Math.pow(a, b); }
            @Override public float invoke(float a, float b) { return (float) Math.pow(a, b); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a * Math.pow( a, b - 1  ); }
            @Override public float invoke(float a, float b) { return (float) (a * Math.pow( a, b - 1  )); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return Math.pow( a, b ) * Math.log(a); }
            @Override public float invoke(float a, float b) { return (float) (Math.pow( a, b ) * Math.log(a)); }
        };
    }
}
