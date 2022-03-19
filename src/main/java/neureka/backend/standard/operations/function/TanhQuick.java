package neureka.backend.standard.operations.function;

import neureka.calculus.Function;

public class TanhQuick extends AbstractActivationOperation
{
    public TanhQuick() { super("quick_tanh"); }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    protected String _activationCode() {
        return "output = input / ( 1.0f + fabs( input ) );\n";
    }

    @Override
    protected String _derivationCode() {
        return "output = 1.0f / ( 2.0f * fabs( input ) + input * input + 1.0f );\n";
    }

    @Override protected double _activate(double x) { return x / ( 1d + Math.abs( x ) ); }

    @Override protected float _activate(float x) { return x / ( 1f + Math.abs( x ) ); }

    @Override protected double _derive(double x) { return 1d / ( 2d * Math.abs( x ) + x * x + 1d ); }

    @Override protected float _derive(float x) { return 1f / ( 2f * Math.abs( x ) + x * x + 1f ); }

}
