package neureka.backend.main.operations.function;

import neureka.calculus.Function;

/**
 *  The Self Gated {@link Tanh} Unit is based on the {@link Tanh}
 *  making it an exponentiation based version of the {@link GaSU} function which
 *  is itself based on the {@link Softsign} function
 *  (a computationally cheap non-exponential quasi {@link Tanh}).
 *  Similar a the {@link Softsign} and {@link Tanh} function {@link GaTU}
 *  is 0 centered and caped by -1 and +1.
 */
public class GaTU  extends AbstractActivationOperation
{
    public GaTU() { super( "gatu" ); }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    protected String _activationCode() { return "output = tanh(input*input*input);\n"; }

    @Override
    protected String _derivationCode() {
        return "float x2 = input * input;       \n" +
               "float x3 = x2 * input;          \n" +
               "float temp = 3 * x2;            \n" +
               "float tanh2 = pow(tanh(x3), 2); \n" +
               "output = -temp * tanh2 + temp;  \n";
    }

    @Override protected double _activate( double x ) { return Tanh.tanh(x*x*x); }

    @Override protected float _activate( float x ) { return Tanh.tanh(x*x*x); }

    @Override protected double _derive( double x ) {
        double x2 = x * x;
        double x3 = x2 * x;
        double temp = 3 * x2;
        double tanh2 = Math.pow(Tanh.tanh(x3), 2);
        return -temp * tanh2 + temp;
    }

    @Override protected float _derive( float x ) {
        float x2 = x * x;
        float x3 = x2 * x;
        float temp = 3 * x2;
        float tanh2 = (float) Math.pow(Tanh.tanh(x3), 2);
        return -temp * tanh2 + temp;
    }

}
