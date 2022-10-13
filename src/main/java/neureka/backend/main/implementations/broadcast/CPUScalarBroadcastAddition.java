package neureka.backend.main.implementations.broadcast;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.main.implementations.fun.api.CPUBiFun;
import neureka.devices.host.CPU;

public class CPUScalarBroadcastAddition extends CPUScalarBroadcast
{
    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        assert call.arity() == 3;
        if ( call.getDerivativeIndex() == 0 )
            return Tsr.of( call.input( 1 ).shape(), 1d ).mut().setIsIntermediate( true );
        else if ( call.getDerivativeIndex() == 1 )
            return Tsr.of( call.input( 2 ).shape(), 1d ).mut().setIsIntermediate( true );
        else
            return super.run(call);
    }

    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double  invoke( double a,  double  b ) { return a + b; }
            @Override public float   invoke( float a,   float   b ) { return a + b; }
            @Override public int     invoke( int a,     int     b ) { return a + b; }
            @Override public long    invoke( long a,    long    b ) { return a + b; }
            @Override public char    invoke( char a,    char    b ) { return (char) (((int)a)+((int)b)); }
            @Override public boolean invoke( boolean a, boolean b ) { return a && b; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt0() {
        return new CPUBiFun() {
            @Override public double  invoke( double a,  double  b ) { return 1; }
            @Override public float   invoke( float a,   float   b ) { return 1; }
            @Override public int     invoke( int a,     int     b ) { return 1; }
            @Override public long    invoke( long a,    long    b ) { return 1; }
            @Override public char    invoke( char a,    char    b ) { return (char) 1; }
            @Override public boolean invoke( boolean a, boolean b ) { return true; }
        };
    }

    @Override
    protected CPUBiFun _getDeriveAt1() {
        return _getDeriveAt0();
    }
}
