package neureka.backend.standard.operations.function;

import neureka.backend.standard.operations.function.internal.FastFun;
import neureka.calculus.Function;

public class TanhFast extends AbstractActivationOperation
{
    public TanhFast() { super("fast_tanh" ); }

    @Override
    public String asDerivative( Function[] children, int derivationIndex ) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    protected String _activationCode() {
        return "output = input * fast_inverse_sqrt( 1.0f + input * input );\n";
    }

    @Override
    protected String _derivationCode() {
        return "float temp1 = input * input;\n" +
                "float temp2 = sqrt( 1 + temp1 );\n" +
                "output = 1 / ( temp1 * temp2 + temp2 );\n";
    }

    @Override protected double _activate(double x) { return x * FastFun.invSqrt( 1d + x * x ); }

    @Override protected float _activate(float x) { return x * FastFun.invSqrt( 1f + x * x ); }

    @Override
    protected double _derive( double x ) {
        double temp1 = x * x;
        double temp2 = Math.sqrt( 1 + temp1 );
        return 1 / ( temp1 * temp2 + temp2 );
    }

    @Override
    protected float _derive( float x ) {
        float temp1 = x * x;
        float temp2 = (float) Math.sqrt( 1 + temp1 );
        return 1 / ( temp1 * temp2 + temp2 );
    }

}
