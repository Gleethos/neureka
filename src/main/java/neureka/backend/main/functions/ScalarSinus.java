package neureka.backend.main.functions;

public final class ScalarSinus implements ScalarFun
{
    @Override public String id() { return "sin"; }

    @Override public String activationCode() { return "output = sin( input );\n"; }

    @Override public String derivationCode() { return "output = cos( input );\n"; }

    @Override public double activate(double x) { return Math.sin(x); }

    @Override public double derive(double x) { return Math.cos(x); }

}
