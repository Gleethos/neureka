package neureka.backend.standard.operations.function;

import neureka.calculus.Function;

/**
 *  The softsign function, defined as {@code x / ( 1 + Math.abs( x ) )},
 *  is a computationally cheap 0 centered activation function
 *  which rescales the inputs between -1 and 1, very much like the {@link Tanh} function.
 *  The softsign function converges polynomially and is computationally cheaper than the
 *  tanh function which converges exponentially.
 *  This makes this function a computationally cheap non-exponential quasi {@link Tanh}!
 */
public class Softsign extends AbstractActivationOperation
{
    public Softsign() { super("softsign"); }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    protected String _activationCode() {
        return "output = input / ( 1.0f + fabs( input ) );\n";
    }

    @Override
    protected String _derivationCode() {
        return "output = 1.0f / ( 2.0f * fabs( input ) + input * input + 1.0f );\n";
    }

    @Override protected double _activate(double x) { return softsign(x); }

    @Override protected float _activate(float x) { return softsign(x); }

    @Override protected double _derive(double x) { return 1d / ( 2d * Math.abs( x ) + x * x + 1d ); }

    @Override protected float _derive(float x) { return 1f / ( 2f * Math.abs( x ) + x * x + 1f ); }

    public static double softsign(double x) { return x / ( 1d + Math.abs( x ) ); }

    public static float softsign(float x) { return x / ( 1f + Math.abs( x ) ); }

}
