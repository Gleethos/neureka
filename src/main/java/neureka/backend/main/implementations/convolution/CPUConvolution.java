package neureka.backend.main.implementations.convolution;

import neureka.backend.main.implementations.fun.api.CPUBiFun;

public class CPUConvolution extends AbstractCPUConvolution
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
