package neureka.backend.main.operations.function;

import neureka.calculus.Function;

public final class Tanh extends AbstractActivationOperation
{
    public Tanh() { super( "tanh" ); }

    @Override
    protected String _activationCode() {
        return "output = tanh(input);\n";
    }

    @Override
    protected String _derivationCode() {
        return "output = 1 - pow( tanh(input), 2.0f );\n";
    }

    @Override protected double _activate(double x ) { return 2 / ( 1 + Math.exp( -x * 2 ) ) - 1; }

    @Override protected float _activate( float x ) { return (float) (2 / ( 1 + Math.exp( -x * 2 ) ) - 1); }

    @Override protected double _derive( double x ) { return  1 - Math.pow( tanh( x ), 2 ); }

    @Override protected float _derive( float x ) { return (float) (1 - Math.pow( tanh( x ), 2 )); }

    public static double tanh( double x ) { return 2 / ( 1 + Math.exp( -x * 2 ) ) - 1; }

    public static float tanh( float x ) { return (float) (2 / ( 1 + Math.exp( -x * 2 ) ) - 1); }

}

