package neureka.backend.main.implementations.broadcast;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.main.implementations.fun.api.CPUBiFun;
import neureka.devices.host.CPU;

public class CPUScalarBroadcastMultiplication extends CPUScalarBroadcast
{
    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        assert call.arity() == 3;
        if ( call.getDerivativeIndex() == 0 )
            return call.input( 2 ).shallowCopy().getMut().setIsIntermediate( true );
        else if ( call.getDerivativeIndex() == 1 )
            return call.input( 1 ).shallowCopy().getMut().setIsIntermediate( true );
        else
            return super.run(call);
    }

    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return a * b; }
            @Override public float   invoke(float a, float b) { return a * b; }
            @Override public int     invoke(int a, int b) { return a * b; }
            @Override public long    invoke(long a, long b) { return a * b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return b; }
            @Override public float   invoke(float a, float b) { return b; }
            @Override public int     invoke(int a, int b) { return b; }
            @Override public long    invoke(long a, long b) { return b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return new CPUBiFun() {
            @Override public double  invoke(double a, double b) { return a; }
            @Override public float   invoke(float a, float b) { return a; }
            @Override public int     invoke(int a, int b) { return a; }
            @Override public long    invoke(long a, long b) { return a; }
        };
    }
}
