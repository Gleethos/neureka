package neureka.backend.main.operations.function.internal;

/**
 *  The GELU activation function is based on the standard Gaussian cumulative distribution function
 *  and is defined as {@code x Φ( x )} and implemented as {@code x * sigmoid(x * 1.702)}.
 *  The GELU non-linearity weighs inputs by their percentile,
 *  rather than gates inputs by their sign as in ReLUs.
 *  Consequently, the GELU can be thought of as a smoother ReLU.
 */
public class CPUGeLU implements ActivationFun
{
    private static final double MOD_F64 = 1.702;
    private static final float  MOD_F32 = 1.702f;

    @Override public String id() { return "gelu"; }

    @Override public String activationCode() { return "output = input / ( 1 + (float) exp(-input * 1.702) );\n"; }

    @Override public String derivationCode() {
        return "float sig = 1.0f / ( 1.0f + exp( -input * 1.702f ) );" +
               "float ds = sig * ( 1.0f - sig );" +
               "output = sig + ds * input * 1.702;\n";
    }

    @Override public double activate(double x) { return gelu(x); }

    @Override public float activate(float x) { return (float) gelu(x); }

    @Override
    public double derive(double x) {
        double sig = CPUSigmoid.sig(x * MOD_F64);
        double ds = sig * ( 1d - sig );
        return sig + ds * x * MOD_F64;
    }

    @Override
    public float derive(float x) {
        float sig = (float) CPUSigmoid.sig(x * MOD_F64);
        float ds = sig * ( 1f - sig );
        return sig + ds * x * MOD_F32;
    }

    public static double gelu(double x) { return x * CPUSigmoid.sig(x * 1.702); }

}
