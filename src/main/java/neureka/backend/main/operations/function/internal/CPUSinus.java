package neureka.backend.main.operations.function.internal;

public final class CPUSinus implements ActivationFun
{
    @Override public String id() { return "sin"; }

    @Override public String activationCode() { return "output = sin( input );\n"; }

    @Override public String derivationCode() { return "output = cos( input );\n"; }

    @Override public double activate(double x) { return Math.sin(x); }

    @Override public double derive(double x) { return Math.cos(x); }

}
