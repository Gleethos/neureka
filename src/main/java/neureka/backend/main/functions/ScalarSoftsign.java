package neureka.backend.main.functions;

/**
 *  The softsign function, defined as {@code x / ( 1 + Math.abs( x ) )},
 *  is a computationally cheap 0 centered activation function
 *  which rescales the inputs between -1 and 1, very much like the {@link ScalarTanh} function.
 *  The softsign function converges polynomially and is computationally cheaper than the
 *  tanh function which converges exponentially.
 *  This makes this function a computationally cheap non-exponential quasi {@link ScalarTanh}!
 */
public class ScalarSoftsign implements ScalarFun
{
    @Override public String id() { return "softsign"; }

    @Override
    public String activationCode() {
        return "output = input / ( 1.0f + fabs( input ) );\n";
    }

    @Override
    public String derivationCode() {
        return "output = 1.0f / ( 2.0f * fabs( input ) + input * input + 1.0f );\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double activate(double x) { return softsign(x); }
            @Override public float activate(float x) { return softsign(x); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double activate(double x) { return 1d / ( 2d * Math.abs( x ) + x * x + 1d ); }
            @Override public float activate(float x) { return 1f / ( 2f * Math.abs( x ) + x * x + 1f ); }
        };
    }

    public static double softsign(double x) { return x / ( 1d + Math.abs( x ) ); }

    public static float softsign(float x) { return x / ( 1f + Math.abs( x ) ); }

}
