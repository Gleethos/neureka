package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.functions.CPUBiFun;

public class CPUBroadcastMultiplication extends CPUBroadcast
{
    public CPUBroadcastMultiplication() {}

    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a * b; }
            @Override public float invoke(float a, float b) { return a * b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return b; }
            @Override public float invoke(float a, float b) { return b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a; }
            @Override public float invoke(float a, float b) { return a; }
        };
    }
}
