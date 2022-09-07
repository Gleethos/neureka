package neureka.backend.main.operations.function.internal;

public final class CPUGaussian implements ActivationFun
{
    @Override public String id() { return "gaus"; }

    @Override public String activationCode() { return "output = exp( -( input * input ) );\n"; }

    @Override public String derivationCode() { return "output = -2 * input * exp( -( input * input ) );\n"; }

    @Override public double activate(double x) { return Math.exp( -( x * x ) ); }

    @Override public double derive(double x) { return -2 * x * Math.exp( -( x * x ) ); }

}
