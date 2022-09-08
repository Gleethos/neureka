package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

/**
 *  The SiLu activation function, also known as the swish function, is defined as {@code x * sigmoid(x)}.
 *  It is a smooth, non-monotonic function that consistently matches
 *  or outperforms ReLU on deep networks,
 *  it is unbounded above and bounded below.
 */
public class ScalarSiLU implements ScalarFun
{
    @Override public String id() { return "silu"; }

    @Override public String activationCode() { return "output = input / ( 1 + (float) exp(-input) );\n"; }

    @Override public String derivationCode() {
        return "float sig = 1.0f / ( 1.0f + exp( -input ) );" +
               "output = sig + ( input * sig * ( 1.0f - sig ) );\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return silu(x); }
            @Override public float invoke(float x) { return (float) silu(x); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override
            public double invoke(double x) {
                double sig = ScalarSigmoid.sig(x);
                return sig + ( x * sig * ( 1d - sig ) );
            }
            @Override
            public float invoke(float x) {
                float sig = (float) ScalarSigmoid.sig(x);
                return sig + ( x * sig * ( 1f - sig ) );
            }
        };
    }


    public static double silu(double x) { return x * ScalarSigmoid.sig(x); }

}
