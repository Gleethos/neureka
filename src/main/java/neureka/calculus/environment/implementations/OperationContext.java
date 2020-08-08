package neureka.calculus.environment.implementations;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.function.*;
import neureka.calculus.environment.implementations.indexer.Product;
import neureka.calculus.environment.implementations.indexer.Summation;
import neureka.calculus.environment.implementations.operator.*;
import neureka.calculus.environment.implementations.other.CopyLeft;
import neureka.calculus.environment.implementations.other.CopyRight;
import neureka.calculus.environment.implementations.other.Reshape;

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
    private static final OperationContext _INSTANCE;
    static {
        _INSTANCE = new OperationContext();
        new ReLU();
        new Sigmoid();
        new Tanh();
        new Quadratic();
        new Ligmoid();
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

        new Reshape();
        new CopyLeft();
        new CopyRight();
    }

    /**
     * @return The OperationContext singleton instance!
     */
    public static OperationContext instance(){
        return _INSTANCE;
    }

    private final Map<String, OperationType> _LOOKUP;
    private final ArrayList<OperationType> _REGISTER;
    private int _ID;

    private OperationContext(){
        _LOOKUP = new HashMap<>();
        _REGISTER = new ArrayList<>();
        _ID = 0;
    }

    /**
     * @return A mapping between OperationType identifiers and their corresponding instances.
     */
    public Map<String, OperationType> getLookup(){
        return _LOOKUP;
    }

    /**
     * @return A list of all OperationType instances.
     */
    public List<OperationType> getRegister(){
        return _REGISTER;
    }

    /**
     * @return The ID of the OperationType that will be instantiated next.
     */
    public int getID(){
        return _ID;
    }

    public void incrementID(){
        _ID++;
    }

    public List<OperationType> instances(){
        return getRegister();
    }

    public OperationType instance(int index){
        return getRegister().get(index);
    }

    public OperationType instance(String identifier){
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
