package neureka.backend.main.operations.function;

import neureka.calculus.Function;

public final class Sigmoid extends AbstractActivationOperation
{
    public Sigmoid() { super( "sig" ); }

    @Override protected String _activationCode() { return "output = 1 / ( 1 + (float) exp(-input) );\n"; }

    @Override protected String _derivationCode() { return "output = input * ( 1 - input );\n"; }

    @Override protected double _activate(double x) { return sig(x); }

    @Override protected float _activate(float x) { return (float) sig(x); }

    @Override
    protected double _derive(double x) {
        double sig = _activate(x);
        return sig * ( 1 - sig );
    }

    @Override
    protected float _derive(float x) {
        float sig = _activate(x);
        return sig * ( 1 - sig );
    }

    public static double sig(double x) { return 1d / ( 1d + Math.exp( -x ) ); }

}
