package neureka.backend.main.operations.function.scalar;

public final class ScalarLogarithm implements ScalarFun
{
    @Override public String id() { return "ln"; }

    @Override
    public String activationCode() { return "output = log( input );\n"; }

    @Override
    public String derivationCode() { return "output = 1.0 / ( input );\n"; }

    @Override public double activate(double x) { return Math.log(x); }

    @Override public double derive(double x) { return 1d / x; }

    @Override public float derive(float x) { return 1f / x; }

}
