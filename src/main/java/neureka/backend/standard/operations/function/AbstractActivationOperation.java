package neureka.backend.standard.operations.function;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.calculus.Function;
import org.jetbrains.annotations.Contract;

abstract class AbstractActivationOperation extends AbstractOperation {

    AbstractActivationOperation(OperationBuilder builder)
    {
        super(builder);
    }

    @Override
    public final String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return getIdentifier() + expression;
        return getIdentifier() + "(" + expression + ")";
    }

    @Override
    public final double calculate( double[] inputs, int j, int d, Function[] src ) {
        return calculate(
                src[ 0 ].call( inputs, j ),
                d >= 0
        ) * ( ( d < 0 ) ? 1 : src[ 0 ].derive( inputs, d, j ) );
    }

    @Contract(pure = true)
    private final double calculate(double input, boolean derive ) {
        if ( !derive )
            return _activate( input );
        else
            return _derive( input ) ;
    }

    protected abstract double _activate(double x);

    protected abstract float _activate(float x);

    protected int _activate(int x) { return (int) Math.round( _activate( (double) x ) ); }

    protected abstract double _derive(double x);

    protected abstract float _derive(float x);

    protected int _derive(int x) { return (int) Math.round( _derive( (double) x ) ); }

}
