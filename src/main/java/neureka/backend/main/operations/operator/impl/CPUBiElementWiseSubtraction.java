package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.functions.CPUBiFun;

public class CPUBiElementWiseSubtraction extends CPUBiElementWise
{
    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a - b; }
            @Override public float invoke(float a, float b) { return a - b; }
            @Override public int invoke(int a, int b) { return a - b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return 1d; }
            @Override public float invoke(float a, float b) { return 1f; }
            @Override public int invoke(int a, int b) { return 1; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return -1d; }
            @Override public float invoke(float a, float b) { return -1f; }
            @Override public int invoke(int a, int b) { return -1; }
        };
    }
}
