package neureka.backend.main.operations.function;

import neureka.calculus.Function;

public final class Absolute extends AbstractActivationOperation
{
    public Absolute() { super( "abs" ); }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected String _activationCode() { return "output = fabs( input );\n"; }

    @Override protected String _derivationCode() { return "output = ( input < 0 ) ? -1 : 1;\n"; }

    @Override protected double _activate(double x) { return Math.abs( x ); }

    @Override protected double _derive(double x) { return ( x < 0d ? -1d : 1d ); }

    @Override protected float _activate(float x) { return Math.abs( x ); }

    @Override protected float _derive(float x) { return ( x < 0f ? -1f : 1f ); }

    @Override protected int _activate(int x) { return Math.abs( x ); }

    @Override protected int _derive(int x) { return ( x < 0 ? -1 : 1 ); }
}
