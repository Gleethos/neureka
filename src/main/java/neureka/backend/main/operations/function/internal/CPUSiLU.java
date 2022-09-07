package neureka.backend.main.operations.function.internal;

/**
 *  The SiLu activation function, also known as the swish function, is defined as {@code x * sigmoid(x)}.
 *  It is a smooth, non-monotonic function that consistently matches
 *  or outperforms ReLU on deep networks,
 *  it is unbounded above and bounded below.
 */
public class CPUSiLU implements ActivationFun
{
    @Override public String id() { return "silu"; }

    @Override public String activationCode() { return "output = input / ( 1 + (float) exp(-input) );\n"; }

    @Override public String derivationCode() {
        return "float sig = 1.0f / ( 1.0f + exp( -input ) );" +
               "output = sig + ( input * sig * ( 1.0f - sig ) );\n";
    }

    @Override public double activate(double x) { return silu(x); }

    @Override public float activate(float x) { return (float) silu(x); }

    @Override
    public double derive(double x) {
        double sig = CPUSigmoid.sig(x);
        return sig + ( x * sig * ( 1d - sig ) );
    }

    @Override
    public float derive(float x) {
        float sig = (float) CPUSigmoid.sig(x);
        return sig + ( x * sig * ( 1f - sig ) );
    }

    public static double silu(double x) { return x * CPUSigmoid.sig(x); }

}
