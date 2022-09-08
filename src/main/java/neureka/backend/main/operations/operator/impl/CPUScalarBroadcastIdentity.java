package neureka.backend.main.operations.operator.impl;

import neureka.backend.main.functions.CPUBiFun;

public class CPUScalarBroadcastIdentity extends CPUScalarBroadcast
{
    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return b; }
            @Override public float   invoke(float a, float b) { return b; }
            @Override public byte    invoke(byte a, byte b) { return b; }
            @Override public int     invoke(int a, int b) { return b; }
            @Override public boolean invoke(boolean a, boolean b) { return b; }
            @Override public char    invoke(char a, char b) { return b; }
            @Override public long    invoke(long a, long b) { return b; }
            @Override public Object  invoke(Object a, Object b) { return b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return _getFun();
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return _getFun();
    }
}
