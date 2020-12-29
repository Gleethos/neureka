package neureka.backend.api.operations;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import java.util.*;

/**
 *    PATTERN : Singleton
 *    (But cloneable for testing purposes!)
 *
 *    This class initializes and stores OperationType instances
 *    in various data structures for fast access and querying. (Mostly used by the FunctionParser)
 *
 *    Every OperationType instance contains a default ThreadLocal reference to
 *    the same OperationContext instance, namely: The _INSTANCE variable as declared below.
 *    During class initialization concrete classes extending the OperationType class
 *    are being instantiated in the static block below.
 *
 */
@Accessors( prefix = {"_"}, fluent = true )
@ToString
@Slf4j
public class OperationContext implements Cloneable
{
    private static final ThreadLocal<OperationContext> _contexts = ThreadLocal.withInitial(
            () -> new OperationContext()
    );

    static
    {
       // loading operations!
       ServiceLoader<Operation> serviceLoader = ServiceLoader.load(Operation.class);
       serviceLoader.reload();
       //checking if load was successful
       for (Operation operation : serviceLoader) {
           assert operation.getFunction() != null;
           log.debug( "Operation: '" + operation.getFunction() + "' loaded!" );
       }
    }

    /**
     * @return The OperationContext singleton instance!
     */
    public static OperationContext get()
    {
        return _contexts.get();
    }

    public static void setInstance( OperationContext context )
    {
        _contexts.set(context);
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
     *  The ID of the OperationType that will be instantiated next.
     */
    @Getter private int _id;

    private OperationContext()
    {
        _lookup = new HashMap<>();
        _instances = new ArrayList<>();
        _id = 0;
    }

    public void incrementID()
    {
        _id++;
    }

    public Operation instance( int index ) {
        return _instances.get(index);
    }

    public Operation instance( String identifier ) {
        return _lookup.getOrDefault(identifier, null);
    }

    @Override
    public OperationContext clone()
    {
        OperationContext clone = new OperationContext();
        clone._id = _id;
        clone._lookup.putAll(_lookup);
        clone._instances.addAll(_instances);
        return clone;
    }

}
