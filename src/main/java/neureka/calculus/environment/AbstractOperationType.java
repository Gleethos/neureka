
package neureka.calculus.environment;

import neureka.calculus.Function;
import neureka.calculus.environment.implementations.AbstractOperationTypeImplementation;
import neureka.calculus.environment.operations.OperationContext;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public abstract class AbstractOperationType implements OperationType
{

    private Stringifier _stringifier;

    protected int _id;
    protected String _function;
    protected String _operator;
    /**
     * Arity is the number of arguments or operands
     * that this function or operation takes.
     */
    protected int _arity;
    protected boolean _isOperator;
    protected boolean _isIndexer;
    protected boolean _isCommutative;
    protected boolean _isAssociative;

    private final Map<Class, AbstractOperationTypeImplementation> _implementations = new LinkedHashMap<>();

    public AbstractOperationType(
            String function,
            String operator,
            int arity,
            boolean isOperator,
            boolean isIndexer,
            boolean isCommutative,
            boolean isAssociative
    ) {
        _function = function;
        _arity = arity;
        _id = OperationContext.instance().getID();
        OperationContext.instance().incrementID();
        _operator = operator;
        _isOperator = isOperator;
        _isIndexer = isIndexer;
        _isCommutative = isCommutative;
        _isAssociative = isAssociative;

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

    public abstract double calculate(double[] inputs, int j, int d, List<Function> src);

    //==================================================================================================================

    @Override
    public <T extends AbstractOperationTypeImplementation> T getImplementation(Class<T> type){
        return (T) _implementations.get(type);
    }
    @Override
    public <T extends AbstractOperationTypeImplementation> boolean supportsImplementation(Class<T> type){
        return _implementations.containsKey(type);
    }
    @Override
    public <T extends AbstractOperationTypeImplementation> OperationType setImplementation(Class<T> type, T instance) {
        _implementations.put(type, instance);
        return this;
    }

    @Override
    public OperationType forEachImplementation(Consumer<OperationTypeImplementation> action ){
        _implementations.values().forEach(action);
        return this;
    }

    //==================================================================================================================

    @Override
    public OperationType setStringifier(Stringifier stringifier) {
        _stringifier = stringifier;
        return this;
    }

    @Override
    public Stringifier getStringifier() {
        return _stringifier;
    }

    //==================================================================================================================

    @Override
    public OperationTypeImplementation implementationOf(ExecutionCall call) {
        for( OperationTypeImplementation te : _implementations.values() ) {
            if ( te.getSuitabilityChecker().canHandle(call) ) return te;
        }
        return null;
    }

    //==================================================================================================================

    @Override
    public String getFunction(){
        return _function;
    }

    @Override
    public String getOperator(){
        return _operator;
    }

    @Override
    public int getId(){
        return _id;
    }

    @Override
    public int getArity(){
        return _arity;
    }

    @Override
    public boolean isOperator() {
        return _isOperator;
    }

    @Override
    public boolean isIndexer(){
        return _isIndexer;
    }

    @Override
    public boolean isCommutative(){
        return  _isCommutative;
    }

    @Override
    public boolean supports(Class implementation) {
        return _implementations.containsKey(implementation);
    }

}
