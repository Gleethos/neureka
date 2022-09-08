package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class ScalarSigmoid implements ScalarFun
{
    @Override public String id() { return "sig"; }

    @Override public String activationCode() { return "output = 1 / ( 1 + (float) exp(-input) );\n"; }

    @Override public String derivationCode() { return "output = input * ( 1 - input );\n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return sig(x); }
            @Override public float invoke(float x) { return (float) sig(x); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override
            public double invoke(double x) {
                double sig = sig(x);
                return sig * ( 1 - sig );
            }
            @Override
            public float invoke(float x) {
                float sig = (float) sig(x);
                return sig * ( 1 - sig );
            }
        };
    }

    public static double sig(double x) { return 1d / ( 1d + Math.exp( -x ) ); }

}
