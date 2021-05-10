package neureka.backend.api.operations;

import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import neureka.backend.api.Operation;
import neureka.calculus.Function;

import java.util.ArrayList;
import java.util.List;

/**
 *  This builder class builds instances of the {@link Operation} interface.
 *  Implementing the {@link Operation} interface manually can result in a lot of boilerplate code.
 *  A builder class is the perfect fit for the {@link Operation} because the interface mostly
 *  defines simple properties... <br>
 *  In order to ensure that all necessary properties have been set the builder keeps track
 *  of the passed parameters. If not all properties have been set, the builder will trow an exception.
 */
@Accessors( prefix = {"_"}, chain = true )
public class OperationBuilder
{
    interface Stringifier
    {
        String stringify( String[] children );
    }

    interface Derivator
    {
        String asDerivative( Function[] children, int d );
    }

    @Getter @Setter private Stringifier _stringifier = null;
    @Getter @Setter private Derivator _derivator = null;
    @Getter @Setter private String _function = null;
    @Getter @Setter private String _operator = null;
    @Getter @Setter private Integer _arity = null;
    @Getter @Setter private Boolean _isOperator = null;
    @Getter @Setter private Boolean _isIndexer = null;
    @Getter @Setter private Boolean _isDifferentiable = null;
    @Getter @Setter private Boolean _isInline = null;
    private boolean _disposed = false;

    public void dispose() {
        _disposed = true;
    }

    public Operation build()
    {
        if ( _disposed ) return null;
        List<String> missing = new ArrayList<>();
        if ( _function == null ) missing.add( "function" );
        if ( _operator == null ) missing.add( "operator" );
        if ( _arity == null ) missing.add( "arity" );
        if ( _isOperator == null ) missing.add( "isOperator" );
        if ( _isIndexer == null ) missing.add( "isIndexer" );
        if ( _isDifferentiable == null ) missing.add( "isDifferentiable" );
        if ( _isInline == null ) missing.add( "isInline" );

        if ( !missing.isEmpty() )
            throw new IllegalStateException("Factory not satisfied! The following properties are missing: '"+ String.join(", ", missing) +"'");
        else
            return new AbstractOperation( this ) {
                @Override
                public String stringify( String[] children ) {
                    return _stringifier.stringify( children );
                }

                @Override
                public String asDerivative( Function[] children, int d ) {
                    return _derivator.asDerivative( children, d );
                }

                @Override
                public double calculate( double[] inputs, int j, int d, Function[] src ) {
                    return src[ 0 ].call( inputs, j );
                }
            };
    }
}



