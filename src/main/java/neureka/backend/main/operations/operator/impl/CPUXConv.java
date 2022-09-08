package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.functions.CPUBiFun;

public class CPUXConv extends CPUConvolution
{
    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return a * b; }
            @Override public float   invoke(float a, float b) { return a * b; }
            @Override public int     invoke(int a, int b) { return a * b; }
            @Override public long    invoke(long a, long b) { return a * b; }
        };
    }

}
