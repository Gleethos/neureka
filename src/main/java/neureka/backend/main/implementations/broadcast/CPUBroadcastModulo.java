package neureka.backend.main.implementations.broadcast;

import neureka.backend.main.implementations.fun.api.CPUBiFun;

public class CPUBroadcastModulo extends CPUBroadcast
{
    public CPUBroadcastModulo() {}

    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a % b; }
            @Override public float invoke(float a, float b) { return a % b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return 1 / b; }
            @Override public float invoke(float a, float b) { return 1 / b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return -(a / Math.pow(b, 2)); }
            @Override public float invoke(float a, float b) { return (float) -(a / Math.pow(b, 2)); }
        };
    }
}
