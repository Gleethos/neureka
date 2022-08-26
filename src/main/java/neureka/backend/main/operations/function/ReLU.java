package neureka.backend.main.operations.function;

import neureka.calculus.Function;

public final class ReLU extends AbstractActivationOperation
{
    public ReLU() { super( "relu" ); }

    @Override
    protected String _activationCode() {
        return "if (input >= 0) {  output = input; } else { output = input * (float)0.01; }\n";
    }

    @Override
    protected String _derivationCode() {
        return "if (input >= 0) { output = (float)1; } else { output = (float)0.01; }\n";
    }

    @Override
    protected double _activate(double x) { return ( x >= 0 ? x : x * .01 ); }

    @Override
    protected float _activate(float x) { return ( x >= 0 ? x : x * .01f ); }

    @Override
    protected double _derive(double x) { return ( x >= 0 ? 1 : .01); }

    @Override
    protected float _derive(float x) { return ( x >= 0 ? 1f : .01f ); }

}
