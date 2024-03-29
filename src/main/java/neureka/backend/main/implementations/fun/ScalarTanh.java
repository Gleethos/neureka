package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class ScalarTanh implements ScalarFun
{
    @Override public String id() { return "tanh"; }

    @Override
    public String activationCode() {
        return "output = tanh(input);\n";
    }

    @Override
    public String derivationCode() {
        return "output = 1 - pow( tanh(input), 2.0f );\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x ) { return 2 / ( 1 + Math.exp( -x * 2 ) ) - 1; }
            @Override public float invoke(float x ) { return (float) (2 / ( 1 + Math.exp( -x * 2 ) ) - 1); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x ) { return  1 - Math.pow( tanh( x ), 2 ); }
            @Override public float invoke(float x ) { return (float) (1 - Math.pow( tanh( x ), 2 )); }
        };
    }

    public static double tanh( double x ) { return 2 / ( 1 + Math.exp( -x * 2 ) ) - 1; }

    public static float tanh( float x ) { return (float) (2 / ( 1 + Math.exp( -x * 2 ) ) - 1); }

}

