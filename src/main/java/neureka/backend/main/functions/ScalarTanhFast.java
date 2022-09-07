package neureka.backend.main.functions;

public class ScalarTanhFast implements ScalarFun
{
    @Override public String id() { return "fast_tanh"; }

    @Override
    public String activationCode() {
        return "output = input * fast_inverse_sqrt( 1.0f + input * input );\n";
    }

    @Override
    public String derivationCode() {
        return "float temp1 = input * input;\n" +
                "float temp2 = sqrt( 1 + temp1 );\n" +
                "output = 1 / ( temp1 * temp2 + temp2 );\n";
    }

    @Override public double activate(double x) { return x * FunUtil.invSqrt( 1d + x * x ); }

    @Override public float activate(float x) { return x * FunUtil.invSqrt( 1f + x * x ); }

    @Override
    public double derive(double x ) {
        double temp1 = x * x;
        double temp2 = Math.sqrt( 1 + temp1 );
        return 1 / ( temp1 * temp2 + temp2 );
    }

    @Override
    public float derive(float x ) {
        float temp1 = x * x;
        float temp2 = (float) Math.sqrt( 1 + temp1 );
        return 1 / ( temp1 * temp2 + temp2 );
    }

}
