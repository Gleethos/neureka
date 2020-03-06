package neureka.calculus.environment;

import neureka.Tsr;
import neureka.acceleration.CPU;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.implementations.function.*;
import neureka.calculus.environment.implementations.indexer.Product;
import neureka.calculus.environment.implementations.indexer.Summation;
import neureka.calculus.environment.implementations.operator.*;
import neureka.calculus.environment.implementations.other.CopyLeft;
import neureka.calculus.environment.implementations.other.CopyRight;
import neureka.calculus.environment.implementations.other.Reshape;
import neureka.calculus.factory.assembly.FunctionBuilder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class OperationType implements Type
{


    private static final Map<String, OperationType> _LOOKUP = new HashMap<>();

    private static final ArrayList<OperationType> _REGISTER = new ArrayList<>();

    public static OperationType LOOKUP(String identifier) {
        return _LOOKUP.getOrDefault(identifier, null);
    }

    public static OperationType instance(int index){
        return _REGISTER.get(index);
    }

    public interface OperationCreator{
        CPU.exec.Operator create(Tsr[] inputs, int d);
    }
    public interface ScalarOperationCreator {
        CPU.exec.Operator create(Tsr[] inputs, double scalar, int d);
    }

    public static OperationType instance(String identifier){
        return _LOOKUP.get(identifier);
    }

    private static int _ID = 0;

    protected int _id = -1;
    protected String _name = "";
    protected String  _identifier = "";
    protected boolean _isFunction = false;
    protected boolean _isOperation = false;
    protected boolean _isIndexer = false;
    protected boolean _isConvection = false;
    protected boolean _isCommutative = false;
    protected boolean _isAssociative = false;

    protected String _activationAsString = "";
    protected String _activationDeriviationAsString = "";
    protected OperationType.OperationCreator _activationOperationCreator = null;

    protected String _scalarOperationAsString = "";
    protected String _scalarDeriviationAsString = "";
    protected OperationType.ScalarOperationCreator _scalarOperationCreator = null;

    protected String _convolveOperationAsString = "";
    protected String _convolveDeriviationAsString = "";
    protected OperationType.OperationCreator _convolveOperationCreator = null;

    protected String _operationAsString = "";
    protected String _operationDeriviationAsString = "";
    protected OperationType.OperationCreator _operationCreator = null;

    static
    {
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

    protected OperationType(
            String name,
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,

            String activationOperationAsString,
            String activationDeriviationAsString,
            OperationCreator activationCreator,

            String scalarOperationAsString,
            String scalarDeriviationAsString,
            ScalarOperationCreator scalarOperationCreator,

            String convolveOperationAsString,
            String convolveDerivativeAsString,
            OperationCreator convolveCreator,

            String operationAsString,
            String operationDeriviationAsString,
            OperationCreator operationCreator
    ) {
        _construct(
                name,
                identifier,
                isFunction,
                isOperation,
                isIndexer,
                isConvection,
                isCommutative,
                isAssociative,

                activationCreator,
                activationOperationAsString,
                activationDeriviationAsString,

                scalarOperationCreator,
                scalarOperationAsString,
                scalarDeriviationAsString,

                convolveOperationAsString,
                convolveDerivativeAsString,
                convolveCreator,

                operationAsString,
                operationDeriviationAsString,
                operationCreator
        );
    }

    public OperationType(
            String name,
            String identifier,
            boolean isFunction,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,

            String activationOperationAsString,
            String activationDeriviationAsString,
            OperationCreator activationCreator,

            String scalarOperationAsString,
            String scalarDeriviationAsString,
            ScalarOperationCreator scalarOperationCreator,

            String convolveOperationAsString,
            String convolveDerivativeAsString,
            OperationCreator convolveCreator,

            String operationAsString,
            String operationDeriviationAsString,
            OperationType.OperationCreator operationCreator
    ) {
        _construct(
                name,
                identifier,
                isFunction,
                !isFunction,
                isIndexer,
                isConvection,
                isCommutative,
                isAssociative,

                activationCreator,
                activationOperationAsString,
                activationDeriviationAsString,

                scalarOperationCreator,
                scalarOperationAsString,
                scalarDeriviationAsString,

                convolveOperationAsString,
                convolveDerivativeAsString,
                convolveCreator,

                operationAsString,
                operationDeriviationAsString,
                operationCreator
        );
    }

    protected void _construct(
            String name,
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,

            OperationType.OperationCreator activationCreator,
            String activationOperationAsString,
            String activationDeriviationAsString,

            OperationType.ScalarOperationCreator scalarOperationCreator,
            String scalarOperationAsString,
            String scalarDeriviationAsString,

            String convolveOperationAsString,
            String convolveDerivativeAsString,
            OperationType.OperationCreator convolveCreator,
            
            String operationAsString,
            String operationDeriviationAsString,
            OperationType.OperationCreator operationCreator
    ) {
        _name = name;
        _id = _ID;
        _ID++;
        _identifier = identifier;
        _isFunction = isFunction;
        _isOperation = isOperation;
        _isIndexer = isIndexer;
        _isConvection = isConvection;
        _isCommutative = isCommutative;
        _isAssociative = isAssociative;
        
        _activationOperationCreator = activationCreator;
        _activationAsString = activationOperationAsString;
        _activationDeriviationAsString = activationDeriviationAsString;
        
        _scalarOperationCreator = scalarOperationCreator;
        _scalarOperationAsString = scalarOperationAsString;
        _scalarDeriviationAsString = scalarDeriviationAsString;

        _convolveOperationAsString = convolveOperationAsString;
        _convolveDeriviationAsString =  convolveDerivativeAsString;
        _convolveOperationCreator = convolveCreator;

        _operationAsString = operationAsString;
        _operationDeriviationAsString = operationDeriviationAsString;
        _operationCreator = operationCreator;
        
        _REGISTER.add(this);
        _LOOKUP.put(identifier, this);
        if(
                identifier
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ){
            if(identifier.contains((""+((char)171)))) {
                System.out.println(identifier.replace((""+((char)171)), "<<"));
                _LOOKUP.put(identifier.replace((""+((char)171)), "<<"), this);
            }
            if(identifier.contains((""+((char)187)))) {
                System.out.println(identifier.replace((""+((char)187)),">>"));
                _LOOKUP.put(identifier.replace((""+((char)187)),">>"), this);
            }
        }
    }

    public static OperationType[] all(){
        return _REGISTER.toArray(new OperationType[0]);
    }

    public static int COUNT(){
        return _ID;
    }

    //==================================================================================================================

    public String getName(){
        return _name;
    }

    //-----------------

    public OperationCreator getActivationCreator(){
        return _activationOperationCreator;
    }

    public String getActivationOperationAsString(){
        return _activationAsString;
    }

    public String getActivationDeriviationAsString(){
        return _activationDeriviationAsString;
    }

    //-----------------

    public ScalarOperationCreator getScalarOperationCreator(){
        return _scalarOperationCreator;
    }

    public String getScalarOperationAsString(){
        return _scalarOperationAsString;
    }

    public String getScalarDeriviationAsString(){
        return _scalarDeriviationAsString;
    }

    //-----------------

    public OperationCreator getConvolveOperationCreator() {
        return _convolveOperationCreator;
    }

    public String getConvolveOperationAsString(){
        return _convolveOperationAsString;
    }

    public String getConvolveDeriviationAsString(){
        return _convolveDeriviationAsString;
    }

    //-----------------

    public OperationCreator getOperationCreator() {
        return _operationCreator;
    }

    public String getOperationAsString(){
        return _operationAsString;
    }

    public String getOperationDeriviationAsString(){
        return _operationDeriviationAsString;
    }

    //==================================================================================================================

    public int id(){
        return _id;
    }

    public String identifier(){
        return _identifier;
    }

    public boolean isOperation(){
        return _isOperation;
    }

    public boolean isFunction(){
        return _isFunction;
    }

    public boolean isIndexer(){
        return _isIndexer;
    }

    public boolean isConvection(){
        return _isConvection;
    }

    public boolean isCommutative(){
        return  _isCommutative;
    }

    public boolean supportsScalar(){
        return !_scalarOperationAsString.equals("") && !_scalarDeriviationAsString.equals("");
    }

    public boolean allowsForward(Tsr[] inputs){
        if (this.isConvection()) return false;
        if (this.identifier().equals(",")) return false; //Reshape
        Tsr last = null;
        for (Tsr t : inputs) {
            if (last!=null) {
                if (!last.shape().equals(t.shape())) {
                    return false;
                }
            }
            last = t;
        }
        return true;
    }

    //@Override
    public ADAgent getADAgentOf(Function f, Tsr[] inputs, int i, boolean forward){

        //Tsr d = (allowsForward(inputs))?f.derive(inputs, i):null;
        Function MUL   = FunctionBuilder.build("(I[0]*I[1])", false);
        Function ADD   = FunctionBuilder.build("(I[0]+I[1])", false);
        Function INV_X = FunctionBuilder.build("I[0]x>>I[1]x>>I[2]", false);

        if(forward){
            Tsr d = f.derive(inputs, i);
            return new ADAgent(
                    ()->d,
                    (t, derivative) -> MUL.activate(new Tsr[]{derivative, d}),
                    null
            );
        } else {
            if(this.identifier().equals(","))
            {
                return new ADAgent(
                        ()->null,
                        (t, derivative) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{derivative},0),
                        (t, error) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{error},0)
                );
            }
            else if (this.isOperation() && !this.isConvection())
            {
                Tsr d = f.derive(inputs, i);
                return new ADAgent(
                        ()->d,
                        (t, derivative) -> MUL.activate(new Tsr[]{derivative, d}),
                        (t, error) -> MUL.activate(new Tsr[]{error, d})
                );
            }
            else if (this.isConvection())
            {
                Tsr d = f.derive(inputs, i);
                return new ADAgent(
                        ()->d,
                        (t, derivative) -> MUL.activate(new Tsr[]{derivative, d}),
                        (t, error) -> INV_X.activate(new Tsr[]{error, d, new Tsr(t.getPayload().shape(), 0)})
                );
            }
        }
        return new ADAgent(
                ()->null,
                (t, derivative) -> null,
                (t, error) -> null
        );

    }


}
