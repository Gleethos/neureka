package neureka.backend.api.operations;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.calculus.Functions;

import java.util.*;
import java.util.function.Supplier;

/**
 *    This class is a (thread-local) Singleton / Multiton managing Operation instances,
 *    which is also cloneable for testing purposes and to enable extending the backend dynamically.
 *    <br><br>
 *    It initializes and stores {@link Operation} instances in various data structures
 *    for fast access and querying (Mostly used by the {@link neureka.calculus.assembly.FunctionParser}).
 *    <br>
 *    Operation instance are always managed by {@link OperationContext}
 *    singleton instances. These contexts are store in the static {@link ThreadLocal} "_CONTEXTS" mapping.
 *    In these context instances {@link Operation}s are stored in simple list and map collections,
 *    namely: <br>
 *    The "_instances" list and the "_lookup" map as declared below.
 *    <br>
 *    <br>
 *    During class initialization concrete classes extending the Operation class
 *    are being instantiated in the static block below via a ServiceLoader.
 *
 */
@Slf4j
@ToString
@Accessors( prefix = {"_"}, fluent = true ) // Getters don't have a "get" prefix for better readability!
public class OperationContext implements Cloneable
{
    private static final ThreadLocal<OperationContext> _CONTEXTS = ThreadLocal.withInitial( OperationContext::new );
    static {
        OperationContext context = OperationContext.get();
        // loading operations!
        ServiceLoader<Operation> serviceLoader = ServiceLoader.load( Operation.class );
        //serviceLoader.reload();

        //checking if load was successful
        for ( Operation operation : serviceLoader ) {
            assert operation.getFunction() != null;
            assert operation.getOperator() != null;
            context.addOperation(operation);
            log.debug( "Operation: '" + operation.getFunction() + "' loaded!" );
        }
    }

    /**
     * @return The OperationContext singleton instance!
     */
    public static OperationContext get()
    {
        return _CONTEXTS.get();
    }

    public Runner runner() {
        return new Runner( this, OperationContext.get() );
    }

    /**
     *  This is a very simple class with a single purpose, namely
     *  it exposes methods which receive lambda instances in order to then execute them
     *  in a given context just to then switch back to the original context again.
     */
    public static class Runner {

        private final OperationContext originalContext;
        private final OperationContext visitedContext;

        private Runner( OperationContext visited, OperationContext originalContext ) {
            this.originalContext = originalContext;
            this.visitedContext = visited;
        }

        public Runner run( Runnable contextSpecificAction ) {
            OperationContext.set( visitedContext );
            contextSpecificAction.run();
            OperationContext.set( originalContext );
            return this;
        }

        public <T> T runAndGet( Supplier<T> contextSpecificAction ) {
            OperationContext.set( visitedContext );
            T result = contextSpecificAction.get();
            OperationContext.set( originalContext );
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
     *  The OperationContext is a thread local singleton.
     *  Therefore, this method will only set the context instance
     *  for the thread which is calling this method.
     *
     * @param context The context which ought to be set as thread local singleton.
     */
    public static void set( OperationContext context )
    {
        _CONTEXTS.set( context );
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

    /**
     *  This static {@link Functions} instance wraps pre-instantiated
     *  {@link Function} instances which are configured to not track their computational history.
     *  This means that no computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the "GraphNode" class... )
     */
    private Functions _getFunction = null;

    public Functions getFunction() {
        if ( _getFunction == null ) _getFunction = new Functions( false );
        return _getFunction;
    }

    private Functions _getAutogradFunction = null;

    public Functions getAutogradFunction() {
        if ( _getAutogradFunction == null ) _getAutogradFunction = new Functions( true );
        return _getAutogradFunction;
    }

    private OperationContext()
    {
        _lookup = new HashMap<>();
        _instances = new ArrayList<>();
        _size = 0;
    }

    public void addOperation( Operation operation ) {
        OperationContext.get().incrementID();
        OperationContext.get().instances().add( operation );
        assert !OperationContext.get().lookup().containsKey( operation.getOperator() );
        assert !OperationContext.get().lookup().containsKey( operation.getFunction() );
        OperationContext.get().lookup().put( operation.getOperator(), operation );
        OperationContext.get().lookup().put( operation.getFunction(), operation );
        OperationContext.get().lookup().put( operation.getOperator().toLowerCase(), operation );
        if (
                operation.getOperator()
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ) {
            if ( operation.getOperator().contains( ""+((char)171) ) )
                this.lookup().put(operation.getOperator().replace((""+((char)171)), "<<"), operation);

            if ( operation.getOperator().contains( ""+((char)187) ) )
                this.lookup().put(operation.getOperator().replace((""+((char)187)),">>"), operation);
        }
    }

    public void incrementID()
    {
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
