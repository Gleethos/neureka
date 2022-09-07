package neureka.backend.main.functions;

public final class ScalarQuadratic implements ScalarFun
{
    @Override public String id() { return "quad"; }

    @Override public String activationCode() { return "output = input * input;\n"; }

    @Override public String derivationCode() { return "output = 2 * input;\n"; }

    @Override public double activate(double x) { return x * x; }

    @Override public double derive(double x) { return 2 * x; }

    @Override public float activate(float x) { return x * x; }

    @Override public float derive(float x) { return 2 * x; }

    @Override public int activate(int x) { return x * x; }

    @Override public int derive(int x) { return 2 * x; }

}
