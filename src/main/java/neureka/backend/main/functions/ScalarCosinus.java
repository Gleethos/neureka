package neureka.backend.main.functions;

public final class ScalarCosinus implements ScalarFun
{
    @Override public String id() { return "cos"; }

    @Override
    public String activationCode() { return "output = cos( input );\n"; }

    @Override
    public String derivationCode() { return "output = -sin( input );\n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override
            public double invoke(double x) { return Math.cos(x); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override
            public double invoke(double x) { return -Math.sin(x); }
        };
    }

}
