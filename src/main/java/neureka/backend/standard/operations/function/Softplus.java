package neureka.backend.standard.operations.function;

import neureka.calculus.Function;


public final class Softplus extends AbstractActivationOperation
{
    public Softplus() { super( "softplus" ); }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    protected String _activationCode() {
        return "output = log( 1.0f + exp( input ) );";
    }

    @Override
    protected String _derivationCode() {
        return "output = 1.0f / ( 1.0f + exp( -input ) );\n";
    }

    @Override protected double _activate(double x) { return Math.log( 1d + Math.exp( x ) ); }

    @Override protected double _derive(double x) { return 1d / ( 1d + Math.exp( -x ) ); }

}
