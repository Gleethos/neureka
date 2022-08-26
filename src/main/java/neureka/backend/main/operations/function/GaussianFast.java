package neureka.backend.main.operations.function;

import neureka.calculus.Function;

public class GaussianFast extends AbstractActivationOperation
{
    public GaussianFast() { super( "fast_gaus" ); }

    @Override protected String _activationCode() { return "output = 1 / ( 1 + input * input );\n"; }

    @Override protected String _derivationCode() {
        return "float x2 = input * input;\n" +
               "output = -2 * input / ( x2 * x2 + 2 * x2 + 1 );\n";
    }

    @Override protected double _activate(double x) { return 1 / ( 1 + x * x ); }

    @Override protected double _derive(double x) {
        double x2 = x * x;
        return  -2 * x / ( x2 * x2 + 2 * x2 + 1 );
    }

    @Override protected float _activate(float x) { return 1 / ( 1 + x * x ); }

    @Override protected float _derive(float x) {
        float x2 = x * x;
        return  -2 * x / ( x2 * x2 + 2 * x2 + 1 );
    }


}
