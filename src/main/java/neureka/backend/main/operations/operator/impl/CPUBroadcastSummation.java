package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.functions.CPUBiFun;

public class CPUBroadcastSummation extends CPUBroadcast
{
    public CPUBroadcastSummation() {}

    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a + b; }
            @Override public float  invoke(float a, float b) { return a + b; }
            @Override public int    invoke(int a, int b) { return a + b; }
            @Override public long   invoke(long a, long b) { return a + b; }
            @Override public char   invoke(char a, char b) { return (char) (((int)a)+((int)b)); }
            @Override public boolean invoke(boolean a, boolean b) { return a && b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return 1; }
            @Override public float  invoke(float a, float b) { return 1; }
            @Override public int    invoke(int a, int b) { return 1; }
            @Override public long   invoke(long a, long b) { return 1; }
            @Override public char   invoke(char a, char b) { return (char) (((int)a)+((int)b)); }
            @Override public boolean invoke(boolean a, boolean b) { return a && b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return 1; }
            @Override public float  invoke(float a, float b) { return 1; }
            @Override public int    invoke(int a, int b) { return 1; }
            @Override public long   invoke(long a, long b) { return 1; }
            @Override public char   invoke(char a, char b) { return (char) (((int)a)+((int)b)); }
            @Override public boolean invoke(boolean a, boolean b) { return a && b; }
        };
    }
}
