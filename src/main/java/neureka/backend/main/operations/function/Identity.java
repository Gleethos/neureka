package neureka.backend.main.operations.function;

import neureka.calculus.Function;

public final class Identity extends AbstractActivationOperation
{
    public Identity() { super( "idy"); }

    @Override protected String _activationCode() { return "output = input; \n"; }

    @Override protected String _derivationCode() { return "output = 1.0f; \n"; }

    @Override protected double _activate(double x) { return x; }

    @Override protected double _derive(double x) { return 1; }

    @Override protected float _activate(float x) { return x; }

    @Override protected float _derive(float x) { return 1; }

    @Override protected int _activate(int x) { return x; }

    @Override protected int _derive(int x) { return 1; }

    @Override protected long _activate(long x) { return x; }

    @Override protected long _derive(long x) { return 1; }

    @Override protected boolean _activate(boolean x) { return x; }

    @Override protected char _activate(char x) { return x; }

    protected Object _activate(Object x) { return x; }

    protected Object _derive(Object x) { return null; }

}
