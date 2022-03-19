package neureka.backend.standard.operations.function;

import neureka.calculus.Function;

public final class Logarithm extends AbstractActivationOperation
{
    public Logarithm() { super( "ln" ); }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        if ( children.length != 1 ) throw new IllegalStateException("Natual logarithm does not support more than 1 argument.");
        return children[0].getDerivative(derivationIndex)+" / "+children[0].toString();
    }

    @Override
    protected String _activationCode() { return "output = log( input );\n"; }

    @Override
    protected String _derivationCode() { return "output = 1.0 / ( input );\n"; }

    @Override protected double _activate(double x) { return Math.log(x); }

    @Override protected double _derive(double x) { return 1d / x; }

    @Override protected float _derive(float x) { return 1f / x; }

}
