package neureka.backend.api;


import neureka.Neureka;
import neureka.backend.api.ini.BackendLoader;
import neureka.backend.api.ini.BackendRegistry;
import neureka.backend.api.ini.ImplementationReceiver;
import neureka.backend.api.ini.LoadingContext;
import neureka.calculus.Function;
import neureka.calculus.FunctionCache;
import neureka.calculus.Functions;
import neureka.calculus.assembly.FunctionParser;
import neureka.calculus.assembly.ParseUtil;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
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
 *    for fast access and querying (Mostly used by the {@link ParseUtil} and {@link FunctionParser}).
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
public final class BackendContext implements Cloneable
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

    private final LazyRef<Functions> _getAutogradFunction;

    /**
     *  This {@link Functions} instance wraps pre-instantiated
     *  {@link Function} instances which are configured to not track their computational history.
     *  This means that no computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the "GraphNode" class... )
     */
    private final LazyRef<Functions> _getFunction;


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
     * This method returns an unmodifiable view of the mapping between the {@link Operation#getIdentifier()} / {@link Operation#getOperator()} properties
     * and the {@link Operation} implementation instances to which they belong.
     * Query operations on the returned map "read through" to the specified map,
     * and attempts to modify the returned map, whether direct or via its collection views,
     * result in an {@link UnsupportedOperationException}.
     *
     * @return An unmodifiable mapping of {@link Operation} properties to the {@link Operation} instances to which they belong.
     */
    public Map<String, Operation> getOperationLookupMap() { return Collections.unmodifiableMap( _lookup ); }

    /**
     * This method returns an unmodifiable view of the
     * list of {@link Operation} implementation instances managed by this context.
     * Query operations on the returned map "read through" to the specified map,
     * and attempts to modify the returned map, whether direct or via its collection views,
     * result in an {@link UnsupportedOperationException}.
     *
     * @return An unmodifiable view of the list of {@link Operation} implementation instances managed by this context
     */
    public List<Operation> getOperations() { return Collections.unmodifiableList( _operations); }

    /**
     * @return The number of {@link Operation} instances stored on this {@link BackendContext}.
     */
    public int size() { return _size; }

    /**
     * @return The {@link Function} and {@link neureka.Tsr} cache of this {@link BackendContext}
     */
    public FunctionCache getFunctionCache() { return _functionCache; }

    /**
     *  This method returns a {@link Functions} instance which wraps pre-instantiated
     *  {@link Function} instances which are configured to not track their computational history.
     *  This means that no computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the {@link neureka.autograd.GraphNode} class... )
     */
    public Functions getFunction() { return _getFunction.get(); }

    /**
     *  This method returns a {@link Functions} instance which wraps pre-instantiated
     *  {@link Function} instances which are configured to track their computational history.
     *  This means that a computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the {@link neureka.autograd.GraphNode} class... )
     *
     * @return A container object which exposes various types of functions with autograd support.
     */
    public Functions getAutogradFunction() { return _getAutogradFunction.get(); }

    /**
     *  This creates a new context which is completely void of any {@link Operation} implementation instances.
     *  Use this constructor to test, debug, build and populate custom execution contexts.
     */
    public BackendContext()
    {
        _getAutogradFunction = LazyRef.of( () -> new Functions( true ) );
        _getFunction = LazyRef.of( () -> new Functions( false ) );
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
        String function = operation.getIdentifier();
        String operator = operation.getOperator();
        assert !_lookup.containsKey( operator );
        assert !_lookup.containsKey( function );
        _lookup.put( operator, operation );
        _lookup.put( function, operation );
        _lookup.put( operator.toLowerCase(), operation );
        _size++;
        return this;
    }

    /**
     * @param operation The {@link Operation} which may or may not be part of this {@link BackendContext}.
     * @return The truth value determining if the provided {@link Operation} is part of this {@link BackendContext}.
     */
    public boolean hasOperation( Operation operation ) {
        return _lookup.containsKey( operation.getIdentifier() );
    }

    /**
     * @param operationIdentifier The {@link Operation} identifier which may be the function name or operator if present.
     * @return The truth value determining if the provided {@link Operation} is part of this {@link BackendContext}.
     */
    public boolean hasOperation( String operationIdentifier ) {
        return _lookup.containsKey( operationIdentifier );
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
    public Operation getOperation( String identifier ) { return _lookup.getOrDefault( identifier, null ); }

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
     *
     * @param extensionClass The type class of the extensions whose presents should be checked.
     * @param <E> The type parameter of the provided type class which requires the type to be an extension.
     * @return The truth value determining if the provided type is present.
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
     * @return A list of all {@link BackendExtension} instances.
     */
    public List<BackendExtension> getExtensions() {
        return extensions.getAll( BackendExtension.class );
    }


    private class Registered<D extends Device<?>> {

        final Class<? extends Operation> operationType;
        final Class<? extends DeviceAlgorithm> algorithmType;
        final Class<? extends D> deviceType;
        final java.util.function.Function<LoadingContext, ImplementationFor<D>> function;

        private Registered(Class<? extends Operation> operationType, Class<? extends DeviceAlgorithm> algorithmType, Class<? extends D> deviceType, java.util.function.Function<LoadingContext, ImplementationFor<D>> function) {
            this.operationType = operationType;
            this.algorithmType = algorithmType;
            this.deviceType = deviceType;
            this.function = function;
        }
    }

    /**
     *  Registers the provided {@link BackendExtension} instance
     *  which can then be accessed via {@link #get(Class)}.
     *
     * @param extension The backend extension component which ought to be stored by this.
     * @return This very {@link BackendContext} instance to allow for method chaining.
     */
    public BackendContext set( BackendExtension extension )
    {
        LogUtil.nullArgCheck( extension, "extension", BackendExtension.class );
        BackendLoader loader = extension.getLoader();
        LogUtil.nullArgCheck( loader, "loader", BackendLoader.class );
        // Now before adding the extension to the backend we first try to load all the implementations:
        List<Registered<?>> registeredList = new ArrayList<>();
        loader.load(BackendRegistry.of(
                new ImplementationReceiver() {
                    @Override
                    public <D extends Device<?>> void accept(
                            Class<? extends Operation> operationType,
                            Class<? extends DeviceAlgorithm> algorithmType,
                            Class<? extends D> deviceType,
                            java.util.function.Function<LoadingContext, ImplementationFor<D>> function
                    ) {
                        registeredList.add(new Registered<>(operationType, algorithmType, deviceType, function));
                    }
                }
        ));
        int count = 0;
        for ( Registered<?> registered : registeredList )
            count += _register( registered ) ? 1 : 0;

        count = registeredList.size() - count;

        if ( count != 0 )
            throw new IllegalStateException(
                "Failed to register "+count+" implementations for extension of type '"+extension.getClass().getSimpleName()+"'."
            );

        extensions.set( extension );
        return this;
    }

    private boolean _register( Registered<?> registered ) {
        for ( Operation o : _operations ) {
            if ( o.getClass().equals( registered.operationType ) ) {
                for ( Algorithm a : o.getAllAlgorithms() ) {
                    // We make sure it is a device algorithm:
                    if ( a instanceof DeviceAlgorithm ) {
                        DeviceAlgorithm da = (DeviceAlgorithm) a;
                        if ( da.getClass().equals( registered.algorithmType ) ) {
                            da.setImplementationFor(
                                registered.deviceType,
                                registered.function.apply(new LoadingContext() {
                                    @Override
                                    public String getAlgorithmName() {
                                        return da.getName();
                                    }
                                    @Override
                                    public String getOperationIdentidier() {
                                        return o.getIdentifier();
                                    }
                                })
                            );
                            return true;
                        }
                    }
                }
            }
        }
        return false;
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
