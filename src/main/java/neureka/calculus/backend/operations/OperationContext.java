package neureka.calculus.backend.operations;

import neureka.calculus.backend.operations.linear.XConv;
import neureka.calculus.backend.operations.function.*;
import neureka.calculus.backend.operations.indexer.Product;
import neureka.calculus.backend.operations.indexer.Summation;
import neureka.calculus.backend.operations.operator.*;
import neureka.calculus.backend.operations.other.CopyLeft;
import neureka.calculus.backend.operations.other.CopyRight;
import neureka.calculus.backend.operations.other.DimTrim;
import neureka.calculus.backend.operations.other.Reshape;

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

    private final Map<String, AbstractOperationType> _LOOKUP;
    private final ArrayList<AbstractOperationType> _REGISTER;
    private int _ID;

    private OperationContext() {
        _LOOKUP = new HashMap<>();
        _REGISTER = new ArrayList<>();
        _ID = 0;
    }

    /**
     * @return A mapping between OperationType identifiers and their corresponding instances.
     */
    public Map<String, AbstractOperationType> getLookup() {
        return _LOOKUP;
    }

    /**
     * @return A list of all OperationType instances.
     */
    public List<AbstractOperationType> getRegister() {
        return _REGISTER;
    }

    /**
     * @return The ID of the OperationType that will be instantiated next.
     */
    public int getID() {
        return _ID;
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
        clone._LOOKUP.putAll(_LOOKUP);
        clone._REGISTER.addAll(_REGISTER);
        return clone;
    }

}
