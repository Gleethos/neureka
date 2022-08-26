package neureka.backend.main.operations.function;

import neureka.calculus.Function;

public final class Quadratic extends AbstractActivationOperation
{
    public Quadratic() { super( "quad" ); }

    @Override protected String _activationCode() { return "output = input * input;\n"; }

    @Override protected String _derivationCode() { return "output = 2 * input;\n"; }

    @Override protected double _activate(double x) { return x * x; }

    @Override protected double _derive(double x) { return 2 * x; }

    @Override protected float _activate(float x) { return x * x; }

    @Override protected float _derive(float x) { return 2 * x; }

    @Override protected int _activate(int x) { return x * x; }

    @Override protected int _derive(int x) { return 2 * x; }

}
