package neureka.backend.main.functions;

/**
 * The Scaled Exponential Linear Unit, or SELU, is an activation
 * function that induces self-normalizing properties.
 * The SELU activation function is implemented as:
 * <i>{@code
 *      if      ( x >  0 ) return SCALE * x;
 *      else if ( x <= 0 ) return SCALE * ALPHA * (Math.exp(x) - 1);
 *      else               return Float.NaN;
 * }</i><br>
 * ...where {@code ALPHA == 1.6733} and {@code SCALE == 1.0507}.
 */
public class ScalarSeLU implements ScalarFun
{
    private static final double ALPHA = 1.6732632423543772848170429916717;
    private static final double SCALE = 1.0507009873554804934193349852946;
    private static final float  ALPHA_F32 = (float) ALPHA;
    private static final float  SCALE_F32 = (float) SCALE;


    @Override public String id() { return "selu"; }

    @Override public String activationCode() {
        return "if      ( input > 0  ) output = "+SCALE_F32+"f * input;\n" +
               "else if ( input <= 0 ) output = "+SCALE_F32+"f * "+ALPHA_F32+"f * (exp(input) - 1.0f);\n" +
               "else                   output = 0.0f;\n";
    }

    @Override public String derivationCode() {
        return "if      ( input >  0 ) output = "+SCALE_F32+"f;\n" +
               "else if ( input <= 0 ) output = "+SCALE_F32+"f * "+ALPHA_F32+"f * exp(input);\n" +
               "else                   output = 1.0f;\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double activate(double x) { return selu(x); }
            @Override public float activate(float x) { return (float) selu(x); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override
            public double activate(double x) {
                if      ( x >  0 ) return SCALE;
                else if ( x <= 0 ) return SCALE * ALPHA * Math.exp(x);
                else               return Double.NaN;
            }

            @Override
            public float activate(float x) {
                if      ( x >  0 ) return SCALE_F32;
                else if ( x <= 0 ) return (float) ( SCALE * ALPHA * Math.exp(x) );
                else               return Float.NaN;
            }
        };
    }


    public static double selu(double x) {
        if      ( x >  0 ) return SCALE * x;
        else if ( x <= 0 ) return SCALE * ALPHA * (Math.exp(x) - 1);
        else               return Float.NaN;
    }

}
