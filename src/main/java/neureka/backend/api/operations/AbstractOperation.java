
package neureka.backend.api.operations;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.algorithms.Algorithm;
import neureka.backend.api.algorithms.GenericAlgorithm;
import neureka.backend.api.ExecutionCall;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;

@NoArgsConstructor
@Accessors( prefix = {"_"}, chain = true )
public abstract class AbstractOperation implements Operation
{
    private static Logger _LOG = LoggerFactory.getLogger( AbstractOperation.class );

    @Getter
    @Setter
    private Stringifier _stringifier;

    @Getter protected int _id;
    @Getter protected String _function;
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
     *  This is the default algorithm for every OperationType extending this class.
     *  It may not fit the purpose of every Operation implementation,
     *  however for most operation types it will provide useful functionality to use.
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

    public AbstractOperation(
            String function,
            String operator,
            int arity,
            boolean isOperator,
            boolean isIndexer,
            boolean isDifferentiable,
            boolean isInline
    ) {
        _function = function;
        _arity = arity;
        _id = OperationContext.instance().getID();
        OperationContext.instance().incrementID();
        _operator = operator;
        _isOperator = isOperator;
        _isIndexer = isIndexer;
        _isDifferentiable = isDifferentiable;
        _isInline = isInline;

        OperationContext.instance().getRegister().add( this );
        OperationContext.instance().getLookup().put( operator, this );
        OperationContext.instance().getLookup().put( operator.toLowerCase(), this );
        if (
                operator
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ) {
            if ( operator.contains( ""+((char)171) ) )
                OperationContext.instance().getLookup().put(operator.replace((""+((char)171)), "<<"), this);

            if ( operator.contains( ""+((char)187) ) )
                OperationContext.instance().getLookup().put(operator.replace((""+((char)187)),">>"), this);

        }

    }

    //==================================================================================================================

    @Override
    public <T extends AbstractFunctionalAlgorithm> T getAlgorithm( Class<T> type ) {
        return (T) _algorithms.get( type );
    }

    @Override
    public <T extends AbstractFunctionalAlgorithm> boolean supportsAlgorithm( Class<T> type ) {
        return _algorithms.containsKey( type );
    }

    @Override
    public <T extends AbstractFunctionalAlgorithm> Operation setAlgorithm( Class<T> type, T instance ) {
        _algorithms.put( type, instance );
        return this;
    }

    @Override
    public Operation forEachAlgorithm( Consumer<Algorithm> action ) {
        _algorithms.values().forEach( action );
        return this;
    }

    //==================================================================================================================

    @Override
    public Algorithm AlgorithmFor( ExecutionCall call ) {
        float bestScore = 0f;
        Algorithm bestImpl = null;
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
    public boolean supports( Class implementation ) {
        return _algorithms.containsKey( implementation );
    }

    @Override
    public boolean isOperator() {
        return _isOperator;
    }


}
