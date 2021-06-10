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
 *    This class is managed by {@link Neureka}, a (thread-local) Singleton / Multiton.
 *    An instance of this {@link OperationContext} class hosts {@link Operation} instances.
 *    The context is also cloneable for testing purposes and to enable extending the backend dynamically.
 *    <br><br>
 *    It initializes and stores {@link Operation} instances in various data structures
 *    for fast access and querying (Mostly used by the {@link neureka.calculus.assembly.FunctionParser}).
 *    <br>
 *    {@link Operation}s are stored in simple list and map collections,
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
    public Runner runner() {
        return new Runner( this, Neureka.get().context());
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

    public OperationContext()
    {
        _lookup = new HashMap<>();
        _instances = new ArrayList<>();
        _size = 0;
    }

    public void addOperation( Operation operation ) {
        incrementID();
        instances().add( operation );
        assert !lookup().containsKey( operation.getOperator() );
        assert !lookup().containsKey( operation.getFunction() );
        lookup().put( operation.getOperator(), operation );
        lookup().put( operation.getFunction(), operation );
        lookup().put( operation.getOperator().toLowerCase(), operation );
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
