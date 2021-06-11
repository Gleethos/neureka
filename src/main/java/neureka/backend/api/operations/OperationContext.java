package neureka.backend.api.operations;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import neureka.Neureka;
import neureka.backend.api.Operation;
import neureka.calculus.Cache;
import neureka.calculus.Function;
import neureka.calculus.Functions;

import java.util.*;
import java.util.function.Supplier;

/**
 *    Instances of this class are execution contexts hosting {@link Operation} instances which receive {@link neureka.Tsr}
 *    instances for execution.
 *    {@link OperationContext}s managed by {@link Neureka}, a (thread-local) Singleton / Multiton.  <br>
 *    Contexts are cloneable for testing purposes and to enable extending the backend dynamically.
 *    A given instance also hosts a reference to a {@link Functions} instance which exposes commonly used
 *    pre-instantiated {@link Function} implementation instances.
 *    <br><br>
 *    The {@link OperationContext} initializes and stores {@link Operation} instances in various data structures
 *    for fast access and querying (Mostly used by the {@link neureka.calculus.assembly.FunctionParser}).
 *    <br>
 *    {@link Operation}s are stored in simple list and map collections,
 *    namely: <br>
 *    The "_instances" list and the "_lookup" map as declared below.
 *    <br>
 *    <br>
 *    During class initialization concrete classes extending the {@link Operation} class
 *    are being instantiated in the static block below via a {@link ServiceLoader}.
 *    {@link OperationContext} instances expose a useful class called {@link Runner},
 *    which performs temporary context switching between the caller's context and this
 *    context during the execution of provided lambdas.
 *
 */
@Slf4j
@ToString
@Accessors( prefix = {"_"}, fluent = true ) // Getters don't have a "get" prefix for better readability!
public class OperationContext implements Cloneable
{
    /**
     *  A {@link Runner} wraps both the called context as well as the context of the caller in order
     *  to perform temporary context switching during the execution of lambdas passed to the {@link Runner}.
     *  After a given lambda was executed successfully, the original context will be restored
     *  in the current thread local {@link Neureka} instance.
     *
     * @return A lambda {@link Runner} which performs temporary context switching between the caller's context and this context.
     */
    public Runner runner() {
        return new Runner( this, Neureka.get().context());
    }

    /**
     *  This is a very simple class with a single purpose, namely
     *  it exposes methods which receive lambda instances in order to then execute them
     *  in a given context just to then switch back to the original context again.
     *  A {@link Runner} wraps both the called context as well as the context of the caller in order
     *  to perform temporary context switching during the execution of lambdas passed to the {@link Runner}.
     *  After a given lambda was executed executed, the original context will be restored
     *  in the current thread local {@link Neureka} instance.
     */
    public static class Runner
    {
        private final OperationContext originalContext;
        private final OperationContext visitedContext;

        private Runner( OperationContext visited, OperationContext originalContext ) {
            if ( visited == originalContext ) log.warn("Context runner encountered two identical contexts!");
            this.originalContext = originalContext;
            this.visitedContext = visited;
        }

        public Runner run( Runnable contextSpecificAction ) {
            Neureka.get().setContext( visitedContext );
            contextSpecificAction.run();
            Neureka.get().setContext( originalContext );
            return this;
        }

        public <T> T runAndGet( Supplier<T> contextSpecificAction ) {
            Neureka.get().setContext( visitedContext );
            T result = contextSpecificAction.get();
            Neureka.get().setContext( originalContext );
            return result;
        }

        public <T> T call( Supplier<T> contextSpecificAction ) {
            return runAndGet( contextSpecificAction );
        }

        public <T> T invoke( Supplier<T> contextSpecificAction ) {
            return call( contextSpecificAction );
        }
    }

    /**
     *  A mapping between OperationType identifiers and their corresponding instances.
     */
    @Getter private final Map<String, Operation> _lookup;

    /**
     *  A list of all OperationType instances.
     */
    @Getter private final List<Operation> _instances;

    /**
     *  The number of operation instances stored in this context.
     */
    @Getter private int _size;

    // Global context and cache:
    @Getter private final Cache _functionCache = new Cache();


    /**
     *  This static {@link Functions} instance wraps pre-instantiated
     *  {@link Function} instances which are configured to not track their computational history.
     *  This means that no computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the "GraphNode" class... )
     */
    private Functions _getFunction = null;

    /**
     *  This method returns a {@link Functions} instance which wraps pre-instantiated
     *  {@link Function} instances which are configured to not track their computational history.
     *  This means that no computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the {@link neureka.autograd.GraphNode} class... )
     */
    public Functions getFunction() {
        if ( _getFunction == null ) _getFunction = new Functions( false );
        return _getFunction;
    }

    private Functions _getAutogradFunction = null;

    /**
     *  This method returns a {@link Functions} instance which wraps pre-instantiated
     *  {@link Function} instances which are configured to track their computational history.
     *  This means that a computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the {@link neureka.autograd.GraphNode} class... )
     */
    public Functions getAutogradFunction() {
        if ( _getAutogradFunction == null ) _getAutogradFunction = new Functions( true );
        return _getAutogradFunction;
    }

    /**
     *  This creates a new context which is completely void of any {@link Operation} implementation instances.
     *  Use this constructor to test, debug, build and populate custom execution contexts.
     */
    public OperationContext()
    {
        _lookup = new HashMap<>();
        _instances = new ArrayList<>();
        _size = 0;
    }

    public void addOperation( Operation operation ) {
        instances().add( operation );
        String function = operation.getFunction();
        String operator = operation.getOperator();
        assert !lookup().containsKey( operator );
        assert !lookup().containsKey( function );
        lookup().put( operator, operation );
        lookup().put( function, operation );
        lookup().put( operator.toLowerCase(), operation );
        if (
                operator
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ) {
            if ( operator.contains( ""+((char)171) ) )
                this.lookup().put(operator.replace((""+((char)171)), "<<"), operation);

            if ( operator.contains( ""+((char)187) ) )
                this.lookup().put(operator.replace((""+((char)187)),">>"), operation);
        }
        _size++;
    }


    /**
     *  This method queries the operations in this OperationContext
     *  by a provided id integer which has to match the the id of an
     *  existing operation.
     *
     * @param id The id of the operation.
     * @return The found Operation instance or null.
     */
    public Operation instance( int id )
    {
        return _instances.get( id );
    }

    /**
     *  This method queries the operations in this OperationContext
     *  by a provided identifier which has to match the name of
     *  an existing operation.
     *
     * @param identifier The operation identifier, aka: its name.
     * @return The requested Operation or null.
     */
    public Operation instance( String identifier )
    {
        return _lookup.getOrDefault( identifier, null );
    }

    @Override
    public OperationContext clone()
    {
        OperationContext clone = new OperationContext();
        clone._size = _size;
        clone._lookup.putAll( _lookup );
        clone._instances.addAll( _instances );
        return clone;
    }

}
