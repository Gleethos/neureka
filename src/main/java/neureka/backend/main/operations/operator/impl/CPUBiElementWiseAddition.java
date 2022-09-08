package neureka.backend.main.operations.operator.impl;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.main.functions.CPUBiFun;
import neureka.devices.host.CPU;

public class CPUBiElementWiseAddition extends CPUBiElementWise
{
    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a + b; }
            @Override public float invoke(float a, float b) { return a + b; }
            @Override public int invoke(int a, int b) { return a + b; }
            @Override public long invoke(long a, long b) { return a + b; }
            @Override public boolean invoke(boolean a, boolean b) { return a && b; }
            @Override public char invoke(char a, char b) { return (char) (((int)a)+((int)b)); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return 1d; }
            @Override public float invoke(float a, float b) { return 1f; }
            @Override public int invoke(int a, int b) { return 1; }
            @Override public long invoke(long a, long b) { return 1; }
            @Override public boolean invoke(boolean a, boolean b) { return true; }
            @Override public char invoke(char a, char b) { return (char) 1; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return _getDeriveAt0();
    }

}
