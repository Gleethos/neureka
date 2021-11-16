package neureka.backend.api;


import neureka.Neureka;
import neureka.calculus.Function;
import neureka.calculus.FunctionCache;
import neureka.calculus.Functions;
import neureka.calculus.assembly.ParseUtil;
import org.slf4j.Logger;

import java.util.*;
import java.util.function.Supplier;

/**
 *    Instances of this class are execution contexts hosting {@link Operation} instances which receive {@link neureka.Tsr}
 *    instances for execution.
 *    {@link BackendContext}s are managed by {@link Neureka}, a (thread-local) Singleton / Multiton library context.<br>
 *    Contexts are cloneable for testing purposes and to enable extending the backend dynamically.
 *    A given instance also hosts a reference to a {@link Functions} instance which exposes commonly used
 *    pre-instantiated {@link Function} implementation instances.
 *    <br><br>
 *    The {@link BackendContext} initializes and stores {@link Operation} instances in various data structures
 *    for fast access and querying (Mostly used by the {@link ParseUtil} and {@link neureka.calculus.assembly.FunctionBuilder}).
 *    <br>
 *    {@link Operation}s are stored in simple list and map collections,
 *    namely: <br>
 *    The "_instances" list and the "_lookup" map as declared below.
 *    <br>
 *    <br>
 *    During class initialization concrete classes extending the {@link Operation} class
 *    are being instantiated in the static block below via a {@link ServiceLoader}.
 *    {@link BackendContext} instances expose a useful class called {@link Runner},
 *    which performs temporary context switching between the caller's context and this
 *    context during the execution of provided lambdas.
 *
 */
public class BackendContext implements Cloneable
{
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(BackendContext.class);

    private final Extensions extensions = new Extensions();

    /**
     *  A mapping between OperationType identifiers and their corresponding instances.
     */
    private final Map<String, Operation> _lookup;

    /**
     *  A list of all OperationType instances.
     */
    private final List<Operation> _operations;

    /**
     *  The number of operation instances stored in this context.
     */
    private int _size;

    // Global context and cache:
    private final FunctionCache _functionCache = new FunctionCache();

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
     *  thread local {@link Neureka} instance through the {@link Neureka#setBackend(BackendContext)}) method.
     *
     * @return A lambda {@link Runner} which performs temporary context switching between the caller's context and this context.
     */
    public Runner runner() { return new Runner( this, Neureka.get().backend() ); }

    /**
     * This method returns an unmodifiable view of the mapping between the {@link Operation#getFunction()} / {@link Operation#getOperator()} properties
     * and the {@link Operation} implementation instances to which they belong.
     * Query operations on the returned map "read through" to the specified map,
     * and attempts to modify the returned map, whether direct or via its collection views,
     * result in an {@link UnsupportedOperationException}.
     *
     * @return An unmodifiable mapping of {@link Operation} properties to the {@link Operation} instances to which they belong.
     */
    public Map<String, Operation> getOperationLookupMap() { return Collections.unmodifiableMap( this._lookup ); }

    /**
     * This method returns an unmodifiable view of the
     * list of {@link Operation} implementation instances managed by this context.
     * Query operations on the returned map "read through" to the specified map,
     * and attempts to modify the returned map, whether direct or via its collection views,
     * result in an {@link UnsupportedOperationException}.
     *
     * @return An unmodifiable view of the list of {@link Operation} implementation instances managed by this context
     */
    public List<Operation> getOperations() { return Collections.unmodifiableList( this._operations); }

    /**
     * @return The number of {@link Operation} instances stored on this {@link BackendContext}.
     */
    public int size() { return this._size; }

    /**
     * @return The {@link Function} and {@link neureka.Tsr} cache of this {@link BackendContext}
     */
    public FunctionCache getFunctionCache() { return this._functionCache; }

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
    public BackendContext()
    {
        _lookup = new HashMap<>();
        _operations = new ArrayList<>();
        _size = 0;
    }

    /**
     *  This method registers {@link Operation} implementation instances in this {@link BackendContext}
     *  which is the thread local execution context receiving and processing {@link neureka.Tsr} instances...         <br><br>
     *
     * @param operation The {@link Operation} instance which ought to be registered as part of this execution context.
     * @return This very context instance to allow for method chaining.
     */
    public BackendContext addOperation(Operation operation )
    {
        _operations.add( operation );
        String function = operation.getFunction();
        String operator = operation.getOperator();
        assert !_lookup.containsKey( operator );
        assert !_lookup.containsKey( function );
        _lookup.put( operator, operation );
        _lookup.put( function, operation );
        _lookup.put( operator.toLowerCase(), operation );
        if (
                operator // TODO: Remove this! Its nonsensical and error prone!!
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
        return this;
    }

    /**
     * @param operation The {@link Operation} which may or may not be part of this {@link BackendContext}.
     * @return The truth value determining if the provided {@link Operation} is part of this {@link BackendContext}.
     */
    public boolean hasOperation( Operation operation ) {
        return this._lookup.containsKey( operation.getFunction() );
    }

    /**
     * @param operationIdentifier The {@link Operation} identifier which may be the function name or operator if present.
     * @return The truth value determining if the provided {@link Operation} is part of this {@link BackendContext}.
     */
    public boolean hasOperation( String operationIdentifier ) {
        return this._lookup.containsKey( operationIdentifier );
    }

    /**
     *  This method queries the operations in this {@link BackendContext}
     *  by a provided index integer targeting an entry in the list of {@link Operation} implementation instances
     *  sitting in this execution context.
     *
     * @param index The index of the operation.
     * @return The found Operation instance or null.
     */
    public Operation getOperation( int index ) { return _operations.get( index ); }

    /**
     *  This method queries the operations in this BackendContext
     *  by a provided identifier which has to match the name of
     *  an existing operation.
     *
     * @param identifier The operation identifier, aka: its name.
     * @return The requested Operation or null.
     */
    public Operation getOperation(String identifier ) { return _lookup.getOrDefault( identifier, null ); }

    /**
     *  This method produces a shallow copy of this {@link BackendContext}.
     *  This is useful for debugging, testing and extending contexts during runtime without side effects!  <br>
     *
     * @return A shallow copy of this operation / execution context.
     */
    @Override
    public BackendContext clone()
    {
        BackendContext clone = new BackendContext();
        clone._size = _size;
        clone._lookup.putAll( _lookup );
        clone._operations.addAll( _operations );
        return clone;
    }

    public String toString() {
        return getClass().getSimpleName()+"[size=" + this.size() + "]";
    }

    /**
     *  Checks if this context has an instance of the provided {@link BackendExtension} type.
     */
    public <E extends BackendExtension> boolean has( Class<E> extensionClass ) {
        return extensions.has( extensionClass );
    }

    /**
     *  Returns an instance of the provided {@link BackendExtension} type
     *  or null if no extension of that type was found.
     */
    public <E extends BackendExtension> E get( Class<E> componentClass ) {
        return extensions.get( componentClass );
    }

    /**
     *  Registers the provided {@link BackendExtension} instance
     *  which can then be accessed via {@link #get(Class)}.
     */
    public BackendContext set( BackendExtension extension ) {
        extensions.set( extension );
        return this;
    }

    /**
     *  This is a very simple class with a single purpose, namely
     *  it exposes methods which receive lambda instances in order to then execute them
     *  in a given {@link BackendContext}, just to then switch back to the original context again.
     *  Switching a context simply means that the {@link BackendContext} which produced this {@link Runner}
     *  will temporarily be set as execution context for the current thread
     *  local {@link Neureka} instance.                                              <br><br>
     *
     *  A {@link Runner} wraps both the called context as well as the context of the caller in order
     *  to perform this temporary context switching throughout the execution of the lambdas passed to the {@link Runner}.
     *  After a given lambda was executed, the original context will be restored in the current thread
     *  local {@link Neureka} instance through the {@link Neureka#setBackend(BackendContext)}) method.
     */
    public static class Runner
    {
        private final BackendContext originalContext;
        private final BackendContext visitedContext;

        private Runner(BackendContext visited, BackendContext originalContext ) {
            if ( visited == originalContext ) log.warn("Context runner encountered two identical contexts!");
            this.originalContext = originalContext;
            this.visitedContext = visited;
        }

        /**
         *  Use this method to supply a lambda which will be executed in the {@link BackendContext}
         *  which produced this very {@link Runner} instance.
         *  After the lambda finished execution successfully the original {@link BackendContext} will
         *  be restored for the current thread local {@link Neureka} instance.
         *
         * @param contextSpecificAction The context specific action which will be execute in the {@link BackendContext} which produced this {@link Runner}.
         * @return This very {@link Runner} instance to enable method chaining.
         */
        public Runner run( Runnable contextSpecificAction ) {
            Neureka.get().setBackend( visitedContext );
            contextSpecificAction.run();
            Neureka.get().setBackend( originalContext );
            return this;
        }

        /**
         *  Use this method to supply a lambda which will be executed in the {@link BackendContext}
         *  which produced this very {@link Runner} instance.
         *  After the lambda finished execution successfully the original {@link BackendContext} will be restored.
         *  This method distinguishes itself from the {@link #run(Runnable)} method because the
         *  lambda supplied to this method is expected to return something.
         *  What may be returned is up to the user, one might want to return the result
         *  of a tensor operation which might be exclusively available in the used context.
         *
         * @param contextSpecificAction The context specific action which will be execute in the {@link BackendContext} which produced this {@link Runner}.
         * @param <T> The return type of the supplied context action which will also be returned by this method.
         * @return The result of the supplied context action.
         */
        public <T> T runAndGet( Supplier<T> contextSpecificAction ) {
            Neureka.get().setBackend( visitedContext );
            T result = contextSpecificAction.get();
            Neureka.get().setBackend( originalContext );
            return result;
        }

        /**
         *  Use this method to supply a lambda which will be executed in the {@link BackendContext}
         *  which produced this very {@link Runner} instance.
         *  After the lambda finished execution successfully the original {@link BackendContext} will be restored.
         *  This method distinguishes itself from the {@link #run(Runnable)} method because the
         *  lambda supplied to this method is expected to return something.                            <br>
         *  What may be returned is up to the user, one might want to return the result
         *  of a tensor operation which might be exclusively available in the used context.
         *  This method is doing the exact same thing as the {@link #runAndGet(Supplier)} method,
         *  however its name is shorter and it can even be omitted entirely when using Groovy.          <br><br>
         *
         * @param contextSpecificAction The context specific action which will be execute in the {@link BackendContext} which produced this {@link Runner}.
         * @param <T> The return type of the supplied context action which will also be returned by this method.
         * @return The result of the supplied context action.
         */
        public <T> T call( Supplier<T> contextSpecificAction ) {
            return runAndGet( contextSpecificAction );
        }

        /**
         *  Use this method to supply a lambda which will be executed in the {@link BackendContext}
         *  which produced this very {@link Runner} instance.
         *  After the lambda finished execution successfully the original {@link BackendContext} will be restored.
         *  This method distinguishes itself from the {@link #run(Runnable)} method because the
         *  lambda supplied to this method is expected to return something.                            <br>
         *  What may be returned is up to the user, one might want to return the result
         *  of a tensor operation which might be exclusively available in the used context.
         *  This method is doing the exact same thing as the {@link #runAndGet(Supplier)} method,
         *  however its name is shorter and it can even be omitted entirely when using Kotlin.          <br><br>
         *
         * @param contextSpecificAction The context specific action which will be execute in the {@link BackendContext} which produced this {@link Runner}.
         * @param <T> The return type of the supplied context action which will also be returned by this method.
         * @return The result of the supplied context action.
         */
        public <T> T invoke( Supplier<T> contextSpecificAction ) {
            return call( contextSpecificAction );
        }
    }

}
