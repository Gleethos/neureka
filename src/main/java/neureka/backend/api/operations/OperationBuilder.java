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
    /**
     *  Concrete {@link Operation} types ought to be representable by a function name.
     *  This property will correspond to the {@link Operation#getFunction()} method.
     */
    @Getter @Setter private String _function = null;

    @Getter @Setter private String _operator = null;
    /**
     *  Arity is the number of arguments or operands that this function or operation takes.
     *  This property will correspond to the {@link Operation#getArity()} method.
     */
    @Getter @Setter private Integer _arity = null;
    /**
     *  An operator is an alternative to a function like "sum()" or "prod()". <br>
     *  Examples would be "+, -, * ..."!
     *
     *  This property will correspond to the {@link Operation#isOperator()} method.
     */
    @Getter @Setter private Boolean _isOperator = null;
    /**
     *  This boolean property tell the {@link Function} implementations that this {@link Operation}
     *  ought to be viewed as something to be indexed.
     *  The {@link Function} will use this information to iterate over all the provided inputs and
     *  then execute the function wile also passing the index to the function AST.
     *  The resulting array will then be available to this {@link Operation} as argument list.
     *  This feature works alongside the {@link Function} implementation found in
     *  {@link neureka.calculus.implementations.FunctionVariable}, which represents an input indexed
     *  by the identifier 'j'!
     *
     */
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
                public String asDerivative( Function[] children, int derivationIndex) {
                    return _derivator.asDerivative( children, derivationIndex);
                }

                @Override
                public double calculate( double[] inputs, int j, int d, Function[] src ) {
                    return src[ 0 ].call( inputs, j );
                }
            };
    }
}



