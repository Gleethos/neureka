package neureka.backend.main.functions;

public final class ScalarQuadratic implements ScalarFun
{
    @Override public String id() { return "quad"; }

    @Override public String activationCode() { return "output = input * input;\n"; }

    @Override public String derivationCode() { return "output = 2 * input;\n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double activate(double x) { return x * x; }
            @Override public float activate(float x) { return x * x; }
            @Override public int activate(int x) { return x * x; }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double activate(double x) { return 2 * x; }
            @Override public float activate(float x) { return 2 * x; }
            @Override public int activate(int x) { return 2 * x; }
        };
    }

}
