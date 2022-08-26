package neureka.backend.main.operations.function;

import neureka.calculus.Function;

/**
 *  The Self Gated {@link Softsign} Unit is based on the {@link Softsign} function
 *  (a computationally cheap non-exponential quasi {@link Tanh})
 *  making it a polynomially based version of the {@link GaTU} function which
 *  is itself based on the {@link Tanh} function.
 *  Similar as the {@link Softsign} and {@link Tanh} function {@link GaSU}
 *  is 0 centered and capped by -1 and +1.
 */
public class GaSU extends AbstractActivationOperation
{
    public GaSU() { super("gasu"); }

    @Override
    protected String _activationCode() {
        return "float cubed = input * input * input;        \n" +
               "output = cubed / ( 1.0f + fabs( cubed ) );  \n";
    }

    @Override
    protected String _derivationCode() {
        return "float x2 = input * input;                                        \n" +
               "float x6 = x2 * x2 * x2;                                         \n" +
               "output = 3.0f * x2 / ( 2.0f * x2 * fabs( input ) + x6 + 1.0f );  \n";
    }

    @Override protected double _activate(double x) { return Softsign.softsign(x*x*x); }

    @Override protected float _activate(float x) { return Softsign.softsign(x*x*x); }

    @Override protected double _derive(double x) {
        double x2 = x*x;
        double x6 = x2*x2*x2;
        return 3d * x2 / ( 2d * x2 * Math.abs( x ) + x6 + 1d );
    }

    @Override protected float _derive(float x) {
        float x2 = x*x;
        float x6 = x2*x2*x2;
        return 3f * x2 / ( 2f * x2 * Math.abs( x ) + x6 + 1f );
    }

}

