package neureka.backend.main.functions;

/**
 *  The Self Gated {@link ScalarSoftsign} Unit is based on the {@link ScalarSoftsign} function
 *  (a computationally cheap non-exponential quasi {@link ScalarTanh})
 *  making it a polynomially based version of the {@link ScalarGaTU} function which
 *  is itself based on the {@link ScalarTanh} function.
 *  Similar as the {@link ScalarSoftsign} and {@link ScalarTanh} function {@link ScalarGaSU}
 *  is 0 centered and capped by -1 and +1.
 */
public class ScalarGaSU implements ScalarFun
{
    @Override public String id() { return "gasu"; }

    @Override
    public String activationCode() {
        return "float cubed = input * input * input;        \n" +
               "output = cubed / ( 1.0f + fabs( cubed ) );  \n";
    }

    @Override
    public String derivationCode() {
        return "float x2 = input * input;                                        \n" +
               "float x6 = x2 * x2 * x2;                                         \n" +
               "output = 3.0f * x2 / ( 2.0f * x2 * fabs( input ) + x6 + 1.0f );  \n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return ScalarSoftsign.softsign(x*x*x); }
            @Override public float invoke(float x) { return ScalarSoftsign.softsign(x*x*x); }

        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) {
                double x2 = x*x;
                double x6 = x2*x2*x2;
                return 3d * x2 / ( 2d * x2 * Math.abs( x ) + x6 + 1d );
            }
            @Override public float invoke(float x) {
                float x2 = x*x;
                float x6 = x2*x2*x2;
                return 3f * x2 / ( 2f * x2 * Math.abs( x ) + x6 + 1f );
            }
        };
    }

}

