package neureka.backend.standard.operations.function;

import neureka.calculus.Function;

public final class Identity extends AbstractActivationOperation
{
    public Identity() { super( "idy"); }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override protected String _activationCode() { return "output = input; \n"; }

    @Override protected String _derivationCode() { return "output = 1.0f; \n"; }

    @Override protected double _activate(double x) { return x; }

    @Override protected double _derive(double x) { return 1; }

    @Override protected float _activate(float x) { return x; }

    @Override protected float _derive(float x) { return 1; }

}
