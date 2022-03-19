package neureka.backend.standard.operations.function;

import neureka.calculus.Function;

public final class Gaussian extends AbstractActivationOperation
{
    public Gaussian() { super( "gaus" ); }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected String _activationCode() { return "output = exp( -( input * input ) );\n"; }

    @Override protected String _derivationCode() { return "output = -2 * input * exp( -( input * input ) );\n"; }

    @Override protected double _activate(double x) { return Math.exp( -( x * x ) ); }

    @Override protected double _derive(double x) { return -2 * x * Math.exp( -( x * x ) ); }

}
