package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public class ScalarLog10 implements ScalarFun
{
    @Override public String id() { return "log10"; }

    @Override
    public String activationCode() {
        return "output = log10( input );\n";
    }

    @Override
    public String derivationCode() {
        return "output = 1.0f / ( input * log( 10.0f ) );\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke( double x ) { return Math.log10( x ); }
            @Override public float invoke( float x ) { return (float) Math.log10( x ); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke( double x ) { return 1 / ( x * Math.log( 10 ) ); }
            @Override public float invoke( float x ) { return 1 / ( x * (float) Math.log( 10 ) ); }
        };
    }

}