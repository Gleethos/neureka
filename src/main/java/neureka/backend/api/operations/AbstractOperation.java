
package neureka.backend.api.operations;


import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.FallbackAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 *  This abstract {@link Operation} implementation is a useful template for creating new operations.
 *  It provides a partial implementation which consists of a simple component system for hosting {@link Algorithm} instances
 *  as well as a set of properties which {@link Operation} implementations are expected to have. <br>
 *  Therefore, the number of properties this class needs to receive is rather large.
 *  In order to instantiate it one has to pass {@link OperationBuilder} instance to the constructor.
 *  Using the factory will make the property configuration as readable as possible. <br>
 *
 */
public abstract class AbstractOperation implements Operation
{
    private static Logger _LOG = LoggerFactory.getLogger( AbstractOperation.class );

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
    protected final String _function;

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
    protected final String _operator;

    /**
     * Arity is the number of arguments or operands
     * that this function or operation takes.
     */
    protected final int _arity;

    /**
     *  This flag determines if this operation is auto-indexing passed input arguments.
     *  Auto-indexing inputs means that for a given array of input arguments
     *  the wrapping Function instance will call its child nodes targeted via an
     *  index incrementally.
     *  The variable 'j' in a Functions expressions containing 'I[j]' will then be
     *  resolved to an actual input for a given indexer...
     */
    protected final boolean _isIndexer;

    /**
     *  Certain operations are not differentiable, meaning they cannot participate
     *  in neither forward- or reverse- mode differentiation.
     *  In order to avoid error prone behaviour trying involve
     *  non- differentiable operations will yield proper exceptions.
     */
    protected final boolean _isDifferentiable;

    /**
     *  Inline operations are operations which change the state of the arguments passed to them.
     *
     */
    protected final boolean _isInline;
    protected final boolean _isOperator;

    private final Map<Class<?>, Algorithm<?>> _algorithms = new LinkedHashMap<>();

    /**
     *  This is the default algorithm for every Operation extending this class.
     *  It may not fit the purpose of every Operation implementation,
     *  however for most operation types it will provide useful functionalities.
     *
     *  The default algorithm assumes an operation that is either a function or operator.
     *  Meaning that it assumes that the operation is also differentiable.
     *  Therefore it contains functionality that goes alongside this assumption,
     *  just to name a few :                                                                                            <br>
     *                                                                                                                  <br>
     *  - An ADAgent supplier returning ADAgent instances capable of performing both forward- and reverse- mode AD.     <br>
     *  - A simple result tensor instantiation implementation.                                                          <br>
     *  - A basic threaded execution based on the AST of a given Function object.                                       <br>
     */
    private final Algorithm<?> _defaultAlgorithm;

    public AbstractOperation( OperationBuilder builder )
    {
        builder.dispose();

        _function         = builder.getFunction();
        _arity            = builder.getArity();
        _operator         = builder.getOperator();
        _isOperator       = builder.getIsOperator();
        _isIndexer        = builder.getIsIndexer();
        _isDifferentiable = builder.getIsDifferentiable();
        _isInline         = builder.getIsInline();
        _defaultAlgorithm = new FallbackAlgorithm( "default", _arity, this );
    }

    //==================================================================================================================

    @Override
    public Algorithm<?>[] getAllAlgorithms() {
        return _algorithms.values().toArray(new Algorithm[0]);
    }


    /**
     *  {@link Operation} implementations embody a component system hosting unique {@link Algorithm} instances.
     *  For a given class implementing the {@link Algorithm} class, there can only be a single
     *  instance of it referenced (aka supported) by a given {@link Operation} instance.
     *  This method ensures this in terms of read access by returning only a single instance or null
     *  based on the provided class instance whose type extends the {@link Algorithm} interface.
     *
     * @param type The class of the type which implements {@link Algorithm} as a key to get an existing instance.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return The instance of the specified type if any exists within this {@link Operation}.
     */
    @Override
    public <T extends Algorithm<T>> T getAlgorithm( Class<T> type ) {
        T found = (T) _algorithms.get( type );
        if ( found == null ) { // Maybe the provided type is a superclass of one of the entries...
            return _algorithms.entrySet()
                                    .stream()
                                    .filter( e -> type.isAssignableFrom( e.getKey() ) )
                                    .map( e -> (T) e.getValue() )
                                    .findFirst()
                                    .orElse( null );
        }
        else
            return found;
    }

    /**
     *  This method checks if this {@link Operation} contains an instance of the
     *  {@link Algorithm} implementation specified via its type class.
     *
     * @param type The class of the type which implements {@link Algorithm}.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return The truth value determining if this {@link Operation} contains an instance of the specified {@link Algorithm} type.
     */
    @Override
    public <T extends Algorithm<T>> boolean supportsAlgorithm( Class<T> type ) {
        return _algorithms.containsKey( type );
    }

    /**
     *  {@link Operation} implementations embody a component system hosting unique {@link Algorithm} instances.
     *  For a given class implementing the {@link Algorithm} class, there can only be a single
     *  instance of it referenced (aka supported) by a given {@link Operation} instance.
     *  This method enables the registration of {@link Algorithm} types in the component system of this {@link Operation}.
     *
     * @param type The class of the type which implements {@link Algorithm} as key for the provided instance.
     * @param instance The instance of the provided type class which ought to be referenced (supported) by this {@link Operation}.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return This very {@link Operation} instance to enable method chaining on it.
     */
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
        for( Algorithm<?> impl : _algorithms.values() ) {
            float currentScore = impl.isSuitableFor( call );
            if ( currentScore > bestScore ) {
                if ( currentScore == 1.0 ) return (Algorithm<T>) impl;
                else {
                    bestScore = currentScore;
                    bestImpl = (Algorithm<T>) impl;
                }
            }
        }

        if ( _defaultAlgorithm.isSuitableFor( call ) > 0.0f ) return (Algorithm<T>) _defaultAlgorithm;

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


    public String getFunction() {
        return this._function;
    }

    public String getOperator() {
        return this._operator;
    }

    public int getArity() {
        return this._arity;
    }

    public boolean isIndexer() {
        return this._isIndexer;
    }

    public boolean isDifferentiable() {
        return this._isDifferentiable;
    }

    public boolean isInline() {
        return this._isInline;
    }

    public Algorithm<?> getDefaultAlgorithm() {
        return this._defaultAlgorithm;
    }
}
