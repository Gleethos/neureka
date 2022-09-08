package neureka.backend.main.functions;

public final class ScalarGaussian implements ScalarFun
{
    @Override public String id() { return "gaus"; }

    @Override public String activationCode() { return "output = exp( -( input * input ) );\n"; }

    @Override public String derivationCode() { return "output = -2 * input * exp( -( input * input ) );\n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double activate(double x) { return Math.exp( -( x * x ) ); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double activate(double x) { return -2 * x * Math.exp( -( x * x ) ); }
        };
    }

}
