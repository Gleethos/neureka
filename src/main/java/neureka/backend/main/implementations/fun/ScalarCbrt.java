package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public class ScalarCbrt implements ScalarFun
{
    @Override public String id() { return "cbrt"; }

    @Override
    public String activationCode() {
        return "output = cbrt( input );\n";
    }

    @Override
    public String derivationCode() {
        return "output = 1 / ( 3 * pow( input, 2.0f / 3.0f ) );\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke( double x ) { return Math.cbrt( x ); }
            @Override public float invoke( float x ) { return (float) Math.cbrt( x ); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke( double x ) { return 1 / ( 3 * Math.pow( x, 2.0 / 3.0 ) ); }
            @Override public float invoke( float x ) { return 1 / ( 3 * (float) Math.pow( x, 2.0f / 3.0f ) ); }
        };
    }

}

