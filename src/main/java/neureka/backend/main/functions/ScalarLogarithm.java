package neureka.backend.main.functions;

public final class ScalarLogarithm implements ScalarFun
{
    @Override public String id() { return "ln"; }

    @Override
    public String activationCode() { return "output = log( input );\n"; }

    @Override
    public String derivationCode() { return "output = 1.0 / ( input );\n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return Math.log(x); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) { return 1d / x; }
            @Override public float invoke(float x) { return 1f / x; }
        };
    }

}
