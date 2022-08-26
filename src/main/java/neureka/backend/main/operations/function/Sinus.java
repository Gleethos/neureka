package neureka.backend.main.operations.function;

import neureka.calculus.Function;

public final class Sinus extends AbstractActivationOperation
{
    public Sinus() { super( "sin" ); }

    @Override protected String _activationCode() { return "output = sin( input );\n"; }

    @Override protected String _derivationCode() { return "output = cos( input );\n"; }

    @Override protected double _activate(double x) { return Math.sin(x); }

    @Override protected double _derive(double x) { return Math.cos(x); }

}
