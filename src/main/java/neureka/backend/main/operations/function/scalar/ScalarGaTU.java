package neureka.backend.main.operations.function.scalar;

/**
 *  The Self Gated {@link ScalarTanh} Unit is based on the {@link ScalarTanh}
 *  making it an exponentiation based version of the {@link ScalarGaSU} function which
 *  is itself based on the {@link ScalarSoftsign} function
 *  (a computationally cheap non-exponential quasi {@link ScalarTanh}).
 *  Similar a the {@link ScalarSoftsign} and {@link ScalarTanh} function {@link ScalarGaTU}
 *  is 0 centered and caped by -1 and +1.
 */
public class ScalarGaTU implements ScalarFun
{
    @Override public String id() { return "gatu"; }

    @Override
    public String activationCode() { return "output = tanh(input*input*input);\n"; }

    @Override
    public String derivationCode() {
        return "float x2 = input * input;       \n" +
               "float x3 = x2 * input;          \n" +
               "float temp = 3 * x2;            \n" +
               "float tanh2 = pow(tanh(x3), 2); \n" +
               "output = -temp * tanh2 + temp;  \n";
    }

    @Override public double activate(double x ) { return ScalarTanh.tanh(x*x*x); }

    @Override public float activate(float x ) { return ScalarTanh.tanh(x*x*x); }

    @Override public double derive(double x ) {
        double x2 = x * x;
        double x3 = x2 * x;
        double temp = 3 * x2;
        double tanh2 = Math.pow(ScalarTanh.tanh(x3), 2);
        return -temp * tanh2 + temp;
    }

    @Override public float derive(float x ) {
        float x2 = x * x;
        float x3 = x2 * x;
        float temp = 3 * x2;
        float tanh2 = (float) Math.pow(ScalarTanh.tanh(x3), 2);
        return -temp * tanh2 + temp;
    }

}
