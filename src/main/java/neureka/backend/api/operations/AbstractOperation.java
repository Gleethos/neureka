
package neureka.backend.api.operations;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.Algorithm;
import neureka.backend.api.algorithms.GenericAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 *  This abstract Operation implementation is a useful template for creating new operations.
 *  It provides a partial implementation which consists of a simple component system for hosting Algorithm instances
 *  as well as a set of properties which Operation implementations are expected to have. <br>
 *  Therefore, the number of properties this class needs to receive is rather large.
 *  In order to instantiate it one has to pass OperationFactory instance to the constructor.
 *  Using the factory will make the property configuration as readable as possible. <br>
 *
 */
@NoArgsConstructor
@Accessors( prefix = {"_"}, chain = true )
public abstract class AbstractOperation implements Operation
{
    private static Logger _LOG = LoggerFactory.getLogger( AbstractOperation.class );

    /**
     *  The id (identification) number of an operation is expected to be unique
     *  for every operation instance.
     *  Besides providing identification it is also <b>the order of operations</b>
     *  which determines <b>the binding strength of operators</b>.
     *  For example the '+' operator binds variables stronger than the '+' operator.
     *  Regular functions like 'sig', 'tanh', ... will not be affected by this id order!
     */
    @Getter protected int _id;
    /**
     *  An operation may have two ways in which it can describe itself as String within a Function AST.
     *  The first one is an operator style of representation and the second one a classical function.
     *  So for the 'Addition' operation the following two representations exist: <br>
     * <ul>
     *      <li> Operator: '+';   Example: 'I[0] + 3 + 5 * I[1]'
     *      <li> Function: 'add'; Example: 'add( I[0], 3, 5*I[1] )'
     * </ul>
     * The following String is the latter way of representing the operation, namely: a functional way.
     */
    @Getter protected String _function;
    /**
     *  An operation may have two ways in which it can describe itself as String within a Function AST.
     *  The first one is an operator style of representation and the second one a classical function.
     *  So for the 'Addition' operation the following two representations exist: <br>
     * <ul>
     *      <li> Operator: '+';   Example: 'I[0] + 3 + 5 * I[1]'
     *      <li> Function: 'add'; Example: 'add( I[0], 3, 5*I[1] )'
     * </ul>
     * The following String is the primer way of representing the operation, namely: as an operator.
     */
    @Getter protected String _operator;
    /**
     * Arity is the number of arguments or operands
     * that this function or operation takes.
     */
    @Getter protected int _arity = -1;
    @Getter protected boolean _isIndexer;
    @Getter protected boolean _isDifferentiable;
    @Getter protected boolean _isInline;
    protected boolean _isOperator;

    private final Map<Class<?>, Algorithm<?>> _algorithms = new LinkedHashMap<>();

    /**
     *  This is the default algorithm for every Operation extending this class.
     *  It may not fit the purpose of every Operation implementation,
     *  however for most operation types it will provide useful functionalities.
     *
     *  The default algorithm assumes an operation that is either a function or operator.
     *  Meaning that it assumes that the operation is also differentiable.
     *  Therefore it contains functionality that goes alongside this assumption,
     *  just to name a few :
     *
     *  - An ADAgent supplier returning ADAgent instances capable of performing both forward- and reverse- mode AD.
     *
     *  - A simple result tensor instantiation implementation.
     *
     *  - A basic threaded execution based on the AST of a given Function object.
     */
    @Getter
    private final Algorithm _defaultAlgorithm = new GenericAlgorithm( "default", _arity, this );

    public AbstractOperation( OperationFactory factory )
    {
        factory.dispose();

        _function = factory.getFunction();
        _arity = factory.getArity();
        _operator = factory.getOperator();
        _isOperator = factory.getIsOperator();
        _isIndexer = factory.getIsIndexer();
        _isDifferentiable = factory.getIsDifferentiable();
        _isInline = factory.getIsInline();

        _id = OperationContext.get().id();
        OperationContext.get().incrementID();

        OperationContext.get().instances().add( this );
        OperationContext.get().lookup().put( _operator, this );
        OperationContext.get().lookup().put( _operator.toLowerCase(), this );
        if (
                _operator
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ) {
            if ( _operator.contains( ""+((char)171) ) )
                OperationContext.get().lookup().put(_operator.replace((""+((char)171)), "<<"), this);

            if ( _operator.contains( ""+((char)187) ) )
                OperationContext.get().lookup().put(_operator.replace((""+((char)187)),">>"), this);
        }
    }

    //==================================================================================================================

    @Override
    public <T extends Algorithm<T>> T getAlgorithm( Class<T> type ) {
        return (T) _algorithms.get( type );
    }

    @Override
    public <T extends Algorithm<T>> boolean supportsAlgorithm( Class<T> type ) {
        return _algorithms.containsKey( type );
    }

    @Override
    public <T extends Algorithm<T>> Operation setAlgorithm( Class<T> type, T instance ) {
        _algorithms.put( type, instance );
        return this;
    }

    //==================================================================================================================

    @Override
    public <T extends Algorithm<T>> Algorithm<T> getAlgorithmFor( ExecutionCall<?> call ) {
        float bestScore = 0f;
        Algorithm<T> bestImpl = null;
        for( Algorithm impl : _algorithms.values() ) {
            float currentScore = impl.isAlgorithmSuitableFor( call );
            if ( currentScore > bestScore ) {
                if ( currentScore == 1.0 ) return impl;
                else {
                    bestScore = currentScore;
                    bestImpl = impl;
                }
            }
        }

        if ( _defaultAlgorithm.isAlgorithmSuitableFor( call ) > 0.0f ) return _defaultAlgorithm;

        if ( bestImpl == null ) {
            String message = "No suitable implementation for execution call '"+call+"' could be found.\n" +
                    "Execution process aborted.";
            _LOG.error( message );
            throw new IllegalStateException( message );
        }
        return bestImpl;
    }

    //==================================================================================================================

    @Override
    public <T extends Algorithm<T>> boolean supports( Class<T> implementation ) {
        return _algorithms.containsKey( implementation );
    }

    @Override
    public boolean isOperator() {
        return _isOperator;
    }


}
