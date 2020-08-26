
package neureka.calculus.environment;

import neureka.calculus.environment.implementations.AbstractOperationTypeImplementation;
import neureka.calculus.environment.operations.OperationContext;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class OperationType implements Type
{
    //private static ThreadLocal<OperationContext> _CONTEXTS;
    //static
    //{
    //    _CONTEXTS = ThreadLocal.withInitial( OperationContext::instance );
    //}

    public static List<OperationType> instances(){
        return OperationContext.instance().getRegister();
    }

    public static OperationType instance( int index ) {
        return OperationContext.instance().getRegister().get(index);
    }

    public static OperationType instance(String identifier){
        return OperationContext.instance().getLookup().getOrDefault( identifier, null );
    }

    private Stringifier _stringifier;

    protected int _id;
    protected String _name;
    protected String _identifier;
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

    public OperationType(
            String name,
            String identifier,
            int arity,
            boolean isOperator,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative
    ) {
        _name = name;
        _arity = arity;
        _id = OperationContext.instance().getID();
        OperationContext.instance().incrementID();
        _identifier = identifier;
        _isOperator = isOperator;
        _isIndexer = isIndexer;
        _isCommutative = isCommutative;
        _isAssociative = isAssociative;

        OperationContext.instance().getRegister().add(this);
        OperationContext.instance().getLookup().put(identifier, this);
        if (
                identifier
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ) {
            if (identifier.contains((""+((char)171)))) {
                OperationContext.instance().getLookup().put(identifier.replace((""+((char)171)), "<<"), this);
            }
            if (identifier.contains((""+((char)187)))) {
                OperationContext.instance().getLookup().put(identifier.replace((""+((char)187)),">>"), this);
            }
        }
    }

    public static OperationType[] ALL(){
        return OperationContext.instance().getRegister().toArray(new OperationType[0]);
    }

    public static int COUNT(){
        return OperationContext.instance().getID();
    }

    //==================================================================================================================

    @Override
    public String getName(){
        return _name;
    }

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
    public <T extends AbstractOperationTypeImplementation> Type setImplementation(Class<T> type, T instance) {
        _implementations.put(type, instance);
        return this;
    }

    //==================================================================================================================

    @Override
    public Type setStringifier(Stringifier stringifier) {
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
            if ( te.getHandleChecker().canHandle(call) ) return te;
        }
        return null;
    }

    //==================================================================================================================

    @Override
    public int id(){
        return _id;
    }

    @Override
    public String identifier(){
        return _identifier;
    }

    @Override
    public int arity(){
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
