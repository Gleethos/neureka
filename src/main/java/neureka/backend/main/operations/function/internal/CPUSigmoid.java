package neureka.backend.main.operations.function.internal;

public final class CPUSigmoid implements ActivationFun
{
    @Override public String id() { return "sig"; }

    @Override public String activationCode() { return "output = 1 / ( 1 + (float) exp(-input) );\n"; }

    @Override public String derivationCode() { return "output = input * ( 1 - input );\n"; }

    @Override public double activate(double x) { return sig(x); }

    @Override public float activate(float x) { return (float) sig(x); }

    @Override
    public double derive(double x) {
        double sig = activate(x);
        return sig * ( 1 - sig );
    }

    @Override
    public float derive(float x) {
        float sig = activate(x);
        return sig * ( 1 - sig );
    }

    public static double sig(double x) { return 1d / ( 1d + Math.exp( -x ) ); }

}
