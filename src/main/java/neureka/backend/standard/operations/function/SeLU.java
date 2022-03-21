package neureka.backend.standard.operations.function;

import neureka.calculus.Function;

public class SeLU extends AbstractActivationOperation
{
    private static final double ALPHA = 1.6732632423543772848170429916717;
    private static final double SCALE = 1.0507009873554804934193349852946;
    private static final float  ALPHA_F32 = (float) ALPHA;
    private static final float  SCALE_F32 = (float) SCALE;

    public SeLU() { super( "selu" ); }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected String _activationCode() {
        return "if      ( input > 0  ) output = "+SCALE_F32+"f * input;\n" +
               "else if ( input <= 0 ) output = "+SCALE_F32+"f * "+ALPHA_F32+"f * (exp(input) - 1.0f);\n" +
               "else                   output = 0.0f;\n";
    }

    @Override protected String _derivationCode() {
        return "if      ( input >  0 ) output = "+SCALE_F32+"f;\n" +
               "else if ( input <= 0 ) output = "+SCALE_F32+"f * "+ALPHA_F32+"f * exp(input);\n" +
               "else                   output = 1.0f;\n";
    }

    @Override protected double _activate(double x) { return selu(x); }

    @Override protected float _activate(float x) { return (float) selu(x); }

    @Override
    protected double _derive(double x) {
        if      ( x >  0 ) return SCALE;
        else if ( x <= 0 ) return SCALE * ALPHA * Math.exp(x);
        else               return Double.NaN;
    }

    @Override
    protected float _derive(float x) {
        if      ( x >  0 ) return SCALE_F32;
        else if ( x <= 0 ) return (float) ( SCALE * ALPHA * Math.exp(x) );
        else               return Float.NaN;
    }

    public static double selu(double x) {
        if      ( x >  0 ) return SCALE * x;
        else if ( x <= 0 ) return SCALE * ALPHA * (Math.exp(x) - 1);
        else               return Float.NaN;
    }

}
