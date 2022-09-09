package neureka.backend.main.implementations.elementwise;

import neureka.backend.api.annotations.Loadable;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.implementations.fun.api.CPUBiFun;
import neureka.backend.main.operations.operator.Division;

@Loadable(operation = Division.class, algorithm = BiElementWise.class)
public class CPUBiElementWiseDivision extends CPUBiElementWise
{
    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a / b; }
            @Override public float invoke(float a, float b) { return a / b; }
            @Override public int invoke(int a, int b) { return a / b; }
            @Override public long invoke(long a, long b) { return a / b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return 1d / b; }
            @Override public float invoke(float a, float b) { return 1f / b; }
            @Override public int invoke(int a, int b) { return (int) Math.round(1d / b); }
            @Override public long invoke(long a, long b) { return Math.round(1d / b); }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return -a / Math.pow( b, 2 ); }
            @Override public float invoke(float a, float b) { return (float) -( a / Math.pow( b, 2 ) ); }
            @Override public int invoke(int a, int b) { return (int) Math.round( -a / Math.pow( b, 2 ) ); }
            @Override public long invoke(long a, long b) { return Math.round( -a / Math.pow( b, 2 ) ); }
        };
    }
}
