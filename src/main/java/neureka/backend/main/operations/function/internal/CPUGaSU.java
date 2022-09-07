package neureka.backend.main.operations.function.internal;

/**
 *  The Self Gated {@link CPUSoftsign} Unit is based on the {@link CPUSoftsign} function
 *  (a computationally cheap non-exponential quasi {@link CPUTanh})
 *  making it a polynomially based version of the {@link CPUGaTU} function which
 *  is itself based on the {@link CPUTanh} function.
 *  Similar as the {@link CPUSoftsign} and {@link CPUTanh} function {@link CPUGaSU}
 *  is 0 centered and capped by -1 and +1.
 */
public class CPUGaSU implements ActivationFun
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

    @Override public double activate(double x) { return CPUSoftsign.softsign(x*x*x); }

    @Override public float activate(float x) { return CPUSoftsign.softsign(x*x*x); }

    @Override public double derive(double x) {
        double x2 = x*x;
        double x6 = x2*x2*x2;
        return 3d * x2 / ( 2d * x2 * Math.abs( x ) + x6 + 1d );
    }

    @Override public float derive(float x) {
        float x2 = x*x;
        float x6 = x2*x2*x2;
        return 3f * x2 / ( 2f * x2 * Math.abs( x ) + x6 + 1f );
    }

}

