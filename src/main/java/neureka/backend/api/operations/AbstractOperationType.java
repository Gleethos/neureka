
package neureka.backend.api.operations;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import neureka.backend.api.implementations.AbstractFunctionalOperationTypeImplementation;
import neureka.backend.api.implementations.GenericImplementation;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.implementations.OperationTypeImplementation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;

@NoArgsConstructor
@Accessors( prefix = {"_"}, chain = true )
public abstract class AbstractOperationType implements OperationType
{
    private static Logger _LOG = LoggerFactory.getLogger( AbstractOperationType.class );

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

    private final Map<Class<?>, OperationTypeImplementation<?>> _implementations = new LinkedHashMap<>();

    /**
     *  This is the default implementation for every OperationType extending this class.
     *  It may not fit the purpose of every OperationType implementation,
     *  however for most types it will provide useful functionality to use.
     *
     *  The default implementation assumes an operation that is either a function or operator.
     *  Meaning that it assumes that the operation is also differentiable.
     *  Therefore it contains functionality that goes alongside this assumption,
     *  just to name a few :
     *
     *  - An ADAgent supplier returning ADAgent instances capable of performing both forwrd- and reverse- mode AD.
     *
     *  - A simple result tensor instantiation implementation.
     *
     *  - A basic threaded execution based on the AST of a given Function object.
     */
    @Getter
    private final OperationTypeImplementation _defaultImplementation = new GenericImplementation( "default", _arity, this );

    public AbstractOperationType(
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

        OperationContext.instance().getRegister().add(this);
        OperationContext.instance().getLookup().put(operator, this);
        OperationContext.instance().getLookup().put(operator.toLowerCase(), this);
        if (
                operator
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ) {
            if (operator.contains((""+((char)171)))) {
                OperationContext.instance().getLookup().put(operator.replace((""+((char)171)), "<<"), this);
            }
            if (operator.contains((""+((char)187)))) {
                OperationContext.instance().getLookup().put(operator.replace((""+((char)187)),">>"), this);
            }
        }

    }

    //==================================================================================================================

    @Override
    public <T extends AbstractFunctionalOperationTypeImplementation> T getImplementation(Class<T> type) {
        return (T) _implementations.get(type);
    }
    @Override
    public <T extends AbstractFunctionalOperationTypeImplementation> boolean supportsImplementation(Class<T> type) {
        return _implementations.containsKey(type);
    }
    @Override
    public <T extends AbstractFunctionalOperationTypeImplementation> OperationType setImplementation(Class<T> type, T instance) {
        _implementations.put(type, instance);
        return this;
    }

    @Override
    public OperationType forEachImplementation( Consumer<OperationTypeImplementation> action ) {
        _implementations.values().forEach(action);
        return this;
    }

    //==================================================================================================================

    @Override
    public OperationTypeImplementation implementationOf( ExecutionCall call ) {
        float bestScore = 0f;
        OperationTypeImplementation bestImpl = null;
        for( OperationTypeImplementation impl : _implementations.values() ) {
            float currentScore = impl.isImplementationSuitableFor( call );
            if ( currentScore > bestScore ) {
                if ( currentScore == 1.0 ) return impl;
                else {
                    bestScore = currentScore;
                    bestImpl = impl;
                }
            }
        }

        if (  _defaultImplementation.isImplementationSuitableFor( call ) > 0.0f ) return _defaultImplementation;

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
        return _implementations.containsKey(implementation);
    }



    @Override
    public boolean isOperator() {
        return _isOperator;
    }


}
