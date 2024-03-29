package neureka.backend.main.implementations.broadcast;

import neureka.backend.main.implementations.fun.api.CPUBiFun;

public class CPUScalarBroadcastModulo extends CPUScalarBroadcast
{
    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return a % b; }
            @Override public float   invoke(float a, float b) { return a % b; }
            @Override public int     invoke(int a, int b) { return a % b; }
            @Override public long    invoke(long a, long b) { return a % b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return 1 / b; }
            @Override public float   invoke(float a, float b) { return 1 / b; }
            @Override public int     invoke(int a, int b) { return (int) Math.round(1d / b); }
            @Override public long    invoke(long a, long b) { return Math.round(1d / b); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return -( a / Math.pow( b, 2 ) ); }
            @Override public float   invoke(float a, float b) { return (float) -( a / Math.pow( b, 2 ) ); }
            @Override public int     invoke(int a, int b) { return (int) Math.round( -a / Math.pow( b, 2 ) ); }
            @Override public long    invoke(long a, long b) { return Math.round( -a / Math.pow( b, 2 ) ); }
        };
    }
}
