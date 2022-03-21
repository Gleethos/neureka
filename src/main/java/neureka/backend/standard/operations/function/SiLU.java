package neureka.backend.standard.operations.function;

import neureka.calculus.Function;

/**
 *  The SiLu activation function, also known as the swish function, is defined as {@code x * sigmoid(x)}.
 *  It is a smooth, non-monotonic function that consistently matches
 *  or outperforms ReLU on deep networks,
 *  it is unbounded above and bounded below.
 */
public class SiLU extends AbstractActivationOperation
{
    public SiLU() { super( "silu" ); }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected String _activationCode() { return "output = input / ( 1 + (float) exp(-input) );\n"; }

    @Override protected String _derivationCode() {
        return "float sig = 1.0f / ( 1.0f + exp( -input ) );" +
               "output = sig + ( input * sig * ( 1.0f - sig ) );\n";
    }

    @Override protected double _activate(double x) { return silu(x); }

    @Override protected float _activate(float x) { return (float) silu(x); }

    @Override
    protected double _derive(double x) {
        double sig = Sigmoid.sig(x);
        return sig + ( x * sig * ( 1d - sig ) );
    }

    @Override
    protected float _derive(float x) {
        float sig = (float) Sigmoid.sig(x);
        return sig + ( x * sig * ( 1f - sig ) );
    }

    public static double silu(double x) { return x * Sigmoid.sig(x); }

}
