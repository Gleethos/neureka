package neureka.backend.standard.operations.function;

import neureka.calculus.Function;

/**
 *  The GELU activation function is based on the standard Gaussian cumulative distribution function
 *  and is defined as {@code x Φ( x )} and implemented as {@code x * sigmoid(x * 1.702)}.
 *  The GELU non-linearity weighs inputs by their percentile,
 *  rather than gates inputs by their sign as in ReLUs ( x 1 x > 0 ).
 *  Consequently, the GELU can be thought of as a smoother ReLU.
 */
public class GeLU extends AbstractActivationOperation
{
    private static final double MOD_F64 = 1.702;
    private static final float  MOD_F32 = 1.702f;

    public GeLU() { super( "gelu" ); }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected String _activationCode() { return "output = input / ( 1 + (float) exp(-input * 1.702) );\n"; }

    @Override protected String _derivationCode() {
        return "float sig = 1.0f / ( 1.0f + exp( -input * 1.702f ) );" +
               "float ds = sig * ( 1.0f - sig );" +
               "output = sig + ds * input * 1.702;\n";
    }

    @Override protected double _activate(double x) { return gelu(x); }

    @Override protected float _activate(float x) { return (float) gelu(x); }

    @Override
    protected double _derive(double x) {
        double sig = Sigmoid.sig(x * MOD_F64);
        double ds = sig * ( 1d - sig );
        return sig + ds * x * MOD_F64;
    }

    @Override
    protected float _derive(float x) {
        float sig = (float) Sigmoid.sig(x * MOD_F64);
        float ds = sig * ( 1f - sig );
        return sig + ds * x * MOD_F32;
    }

    public static double gelu(double x) { return x * Sigmoid.sig(x * 1.702); }

}
