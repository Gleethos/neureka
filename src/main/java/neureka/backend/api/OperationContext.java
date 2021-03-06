package neureka.backend.api;


import neureka.Neureka;
import neureka.calculus.Cache;
import neureka.calculus.Function;
import neureka.calculus.Functions;
import org.slf4j.Logger;

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
public class OperationContext implements Cloneable
{
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(OperationContext.class);
    /**
     *  A mapping between OperationType identifiers and their corresponding instances.
     */
    private final Map<String, Operation> _lookup;

    /**
     *  A list of all OperationType instances.
     */
    private final List<Operation> _instances;

    /**
     *  The number of operation instances stored in this context.
     */
    private int _size;

    // Global context and cache:
    private final Cache _functionCache = new Cache();

    private Functions _getAutogradFunction = null;

    /**
     *  This {@link Functions} instance wraps pre-instantiated
     *  {@link Function} instances which are configured to not track their computational history.
     *  This means that no computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the "GraphNode" class... )
     */
    private Functions _getFunction = null;


    /**
     *  A {@link Runner} wraps both the called context as well as the context of the caller in order
     *  to perform temporary context switching during the execution of lambdas passed to the {@link Runner}.
     *  After a given lambda was executed successfully, the original context will be restored in the current
     *  thread local {@link Neureka} instance through the {@link Neureka#setContext(OperationContext)}) method.
     *
     * @return A lambda {@link Runner} which performs temporary context switching between the caller's context and this context.
     */
    public Runner runner() { return new Runner( this, Neureka.get().context()); }

    /**
     * This method returns an unmodifiable view of the mapping between the {@link Operation#getFunction()} / {@link Operation#getOperator()} properties
     * and the {@link Operation} implementation instances to which they belong.
     * Query operations on the returned map "read through" to the specified map,
     * and attempts to modify the returned map, whether direct or via its collection views,
     * result in an {@link UnsupportedOperationException}.
     *
     * @return An unmodifiable mapping of {@link Operation} properties to the {@link Operation} instances to which they belong.
     */
    public Map<String, Operation> lookup() { return Collections.unmodifiableMap( this._lookup ); }

    /**
     * This method returns an unmodifiable view of the
     * list of {@link Operation} implementation instances managed by this context.
     * Query operations on the returned map "read through" to the specified map,
     * and attempts to modify the returned map, whether direct or via its collection views,
     * result in an {@link UnsupportedOperationException}.
     *
     * @return An unmodifiable view of the list of {@link Operation} implementation instances managed by this context
     */
    public List<Operation> instances() { return Collections.unmodifiableList( this._instances ); }

    /**
     * @return The number of {@link Operation} instances stored on this {@link OperationContext}.
     */
    public int size() { return this._size; }

    /**
     * @return The {@link Function} and {@link neureka.Tsr} cache of this {@link OperationContext}
     */
    public Cache functionCache() { return this._functionCache; }

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

    /**
     *  This method registers {@link Operation} implementation instances in this {@link OperationContext}
     *  which is the thread local execution context receiving and processing {@link neureka.Tsr} instances...         <br><br>
     *
     * @param operation The {@link Operation} instance which ought to be registered as part of this execution context.
     */
    public void addOperation( Operation operation )
    {
        _instances.add( operation );
        String function = operation.getFunction();
        String operator = operation.getOperator();
        assert !_lookup.containsKey( operator );
        assert !_lookup.containsKey( function );
        _lookup.put( operator, operation );
        _lookup.put( function, operation );
        _lookup.put( operator.toLowerCase(), operation );
        if (
                operator
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ) {
            if ( operator.contains( ""+((char)171) ) )
                this._lookup.put(operator.replace((""+((char)171)), "<<"), operation);

            if ( operator.contains( ""+((char)187) ) )
                this._lookup.put(operator.replace((""+((char)187)),">>"), operation);
        }
        _size++;
    }


    /**
     *  This method queries the operations in this {@link OperationContext}
     *  by a provided index integer targeting an entry in the list of {@link Operation} implementation instances
     *  sitting in this execution context.
     *
     * @param index The index of the operation.
     * @return The found Operation instance or null.
     */
    public Operation instance( int index ) { return _instances.get( index ); }

    /**
     *  This method queries the operations in this OperationContext
     *  by a provided identifier which has to match the name of
     *  an existing operation.
     *
     * @param identifier The operation identifier, aka: its name.
     * @return The requested Operation or null.
     */
    public Operation instance( String identifier ) { return _lookup.getOrDefault( identifier, null ); }

    /**
     *  This method produces a shallow copy of this {@link OperationContext}.
     *  This is useful for debugging, testing and extending contexts during runtime without side effects!  <br>
     *
     * @return A shallow copy of this operation / execution context.
     */
    @Override
    public OperationContext clone()
    {
        OperationContext clone = new OperationContext();
        clone._size = _size;
        clone._lookup.putAll( _lookup );
        clone._instances.addAll( _instances );
        return clone;
    }

    public String toString() {
        return "OperationContext(_lookup=" + this.lookup() + ", _instances=" + this.instances() + ", _size=" + this.size() + ", _functionCache=" + this.functionCache() + ", _getAutogradFunction=" + this.getAutogradFunction() + ", _getFunction=" + this.getFunction() + ")";
    }


    /**
     *  This is a very simple class with a single purpose, namely
     *  it exposes methods which receive lambda instances in order to then execute them
     *  in a given {@link OperationContext}, just to then switch back to the original context again.
     *  Switching a context simply means that the {@link OperationContext} which produced this {@link Runner}
     *  will temporarily be set as execution context for the current thread
     *  local {@link Neureka} instance.                                              <br><br>
     *
     *  A {@link Runner} wraps both the called context as well as the context of the caller in order
     *  to perform this temporary context switching throughout the execution of the lambdas passed to the {@link Runner}.
     *  After a given lambda was executed, the original context will be restored in the current thread
     *  local {@link Neureka} instance through the {@link Neureka#setContext(OperationContext)}) method.
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

        /**
         *  Use this method to supply a lambda which will be executed in the {@link OperationContext}
         *  which produced this very {@link Runner} instance.
         *  After the lambda finished execution successfully the original {@link OperationContext} will
         *  be restored for the current thread local {@link Neureka} instance.
         *
         * @param contextSpecificAction The context specific action which will be execute in the {@link OperationContext} which produced this {@link Runner}.
         * @return This very {@link Runner} instance to enable method chaining.
         */
        public Runner run( Runnable contextSpecificAction ) {
            Neureka.get().setContext( visitedContext );
            contextSpecificAction.run();
            Neureka.get().setContext( originalContext );
            return this;
        }

        /**
         *  Use this method to supply a lambda which will be executed in the {@link OperationContext}
         *  which produced this very {@link Runner} instance.
         *  After the lambda finished execution successfully the original {@link OperationContext} will be restored.
         *  This method distinguishes itself from the {@link #run(Runnable)} method because the
         *  lambda supplied to this method is expected to return something.
         *  What may be returned is up to the user, one might want to return the result
         *  of a tensor operation which might be exclusively available in the used context.
         *
         * @param contextSpecificAction The context specific action which will be execute in the {@link OperationContext} which produced this {@link Runner}.
         * @param <T> The return type of the supplied context action which will also be returned by this method.
         * @return The result of the supplied context action.
         */
        public <T> T runAndGet( Supplier<T> contextSpecificAction ) {
            Neureka.get().setContext( visitedContext );
            T result = contextSpecificAction.get();
            Neureka.get().setContext( originalContext );
            return result;
        }

        /**
         *  Use this method to supply a lambda which will be executed in the {@link OperationContext}
         *  which produced this very {@link Runner} instance.
         *  After the lambda finished execution successfully the original {@link OperationContext} will be restored.
         *  This method distinguishes itself from the {@link #run(Runnable)} method because the
         *  lambda supplied to this method is expected to return something.                            <br>
         *  What may be returned is up to the user, one might want to return the result
         *  of a tensor operation which might be exclusively available in the used context.
         *  This method is doing the exact same thing as the {@link #runAndGet(Supplier)} method,
         *  however its name is shorter and it can even be omitted entirely when using Groovy.          <br><br>
         *
         * @param contextSpecificAction The context specific action which will be execute in the {@link OperationContext} which produced this {@link Runner}.
         * @param <T> The return type of the supplied context action which will also be returned by this method.
         * @return The result of the supplied context action.
         */
        public <T> T call( Supplier<T> contextSpecificAction ) {
            return runAndGet( contextSpecificAction );
        }

        /**
         *  Use this method to supply a lambda which will be executed in the {@link OperationContext}
         *  which produced this very {@link Runner} instance.
         *  After the lambda finished execution successfully the original {@link OperationContext} will be restored.
         *  This method distinguishes itself from the {@link #run(Runnable)} method because the
         *  lambda supplied to this method is expected to return something.                            <br>
         *  What may be returned is up to the user, one might want to return the result
         *  of a tensor operation which might be exclusively available in the used context.
         *  This method is doing the exact same thing as the {@link #runAndGet(Supplier)} method,
         *  however its name is shorter and it can even be omitted entirely when using Kotlin.          <br><br>
         *
         * @param contextSpecificAction The context specific action which will be execute in the {@link OperationContext} which produced this {@link Runner}.
         * @param <T> The return type of the supplied context action which will also be returned by this method.
         * @return The result of the supplied context action.
         */
        public <T> T invoke( Supplier<T> contextSpecificAction ) {
            return call( contextSpecificAction );
        }
    }

}
