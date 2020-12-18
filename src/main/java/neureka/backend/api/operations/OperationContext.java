package neureka.backend.api.operations;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import neureka.backend.standard.operations.function.Absolute;
import neureka.backend.standard.operations.function.Cosinus;
import neureka.backend.standard.operations.function.Gaussian;
import neureka.backend.standard.operations.function.Identity;
import neureka.backend.standard.operations.function.Quadratic;
import neureka.backend.standard.operations.function.ReLU;
import neureka.backend.standard.operations.function.Sigmoid;
import neureka.backend.standard.operations.function.Sinus;
import neureka.backend.standard.operations.function.Softplus;
import neureka.backend.standard.operations.function.Tanh;
import neureka.backend.standard.operations.indexer.Product;
import neureka.backend.standard.operations.indexer.Summation;
import neureka.backend.standard.operations.linear.XConv;
import neureka.backend.standard.operations.operator.Addition;
import neureka.backend.standard.operations.operator.Division;
import neureka.backend.standard.operations.operator.Modulo;
import neureka.backend.standard.operations.operator.Multiplication;
import neureka.backend.standard.operations.operator.Power;
import neureka.backend.standard.operations.operator.Subtraction;
import neureka.backend.standard.operations.other.CopyRight;
import neureka.backend.standard.operations.other.DimTrim;
import neureka.backend.standard.operations.other.Reshape;
import neureka.backend.standard.operations.other.CopyLeft;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
@Accessors( prefix = {"_"} )
@ToString
public class OperationContext implements Cloneable
{
    private static ThreadLocal<OperationContext> _INSTANCES = ThreadLocal.withInitial( ()->new OperationContext() );

    static
    {
        new ReLU();
        new Sigmoid();
        new Tanh();
        new Quadratic();
        new Softplus();
        new Identity();
        new Gaussian();
        new Absolute();
        new Sinus();
        new Cosinus();

        new Summation();
        new Product();

        new Power();
        new Division();
        new Multiplication();
        new Modulo();
        new Subtraction();
        new Addition();

        new XConv();

        new Reshape();
        new DimTrim();
        new CopyLeft();
        new CopyRight();
    }

    /**
     * @return The OperationContext singleton instance!
     */
    public static OperationContext instance() {
        return _INSTANCES.get();
    }

    public static void setInstance( OperationContext context ) {
        _INSTANCES.set(context);
    }

    /**
     * @return A mapping between OperationType identifiers and their corresponding instances.
     */
    @Getter private final Map<String, AbstractOperationType> _lookup;
    /**
     * @return A list of all OperationType instances.
     */
    @Getter private final ArrayList<AbstractOperationType> _register;
    /**
     * @return The ID of the OperationType that will be instantiated next.
     */
    @Getter private int _ID;

    private OperationContext() {
        _lookup = new HashMap<>();
        _register = new ArrayList<>();
        _ID = 0;
    }

    public void incrementID() {
        _ID++;
    }

    public List<AbstractOperationType> instances() {
        return getRegister();
    }

    public AbstractOperationType instance(int index) {
        return getRegister().get(index);
    }

    public AbstractOperationType instance(String identifier) {
        return getLookup().getOrDefault(identifier, null);
    }

    @Override
    public OperationContext clone()
    {
        OperationContext clone = new OperationContext();
        clone._ID = _ID;
        clone._lookup.putAll(_lookup);
        clone._register.addAll(_register);
        return clone;
    }

}
