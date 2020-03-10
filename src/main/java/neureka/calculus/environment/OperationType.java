package neureka.calculus.environment;

import neureka.Tsr;
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
import java.util.logging.LoggingPermission;

public class OperationType implements Type
{
    private static final Map<String, OperationType> _LOOKUP = new HashMap<>();

    private static final ArrayList<OperationType> _REGISTER = new ArrayList<>();

    public static OperationType instance(int index){
        return _REGISTER.get(index);
    }

    public static OperationType instance(String identifier){
        return _LOOKUP.getOrDefault(identifier, null);
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

    protected Activation _activation;
    protected Convolution _convolution;
    protected Broadcast _broadcast;
    protected Scalarization _scalarization;
    protected Operation _operation;

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

    public OperationType(
            String name,
            String identifier,
            boolean isFunction,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,

            Activation activation,
            Scalarization scalarization,
            Convolution convolution,
            Broadcast broadcast,
            Operation operation
    ) {
        _construct(
                name, identifier, isFunction, !isFunction, isIndexer, isConvection, isCommutative, isAssociative,
                activation,
                scalarization,
                convolution,
                broadcast,
                operation
        );
    }

    public OperationType(
            String name,
            String identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,

            Activation activation,
            Scalarization scalarization,
            Convolution convolution,
            Broadcast broadcast,
            Operation operation
    ) {
        _construct(
                name, identifier, isFunction, isOperation, isIndexer, isConvection, isCommutative, isAssociative,
                activation,
                scalarization,
                convolution,
                broadcast,
                operation
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

            Activation activation,
            Scalarization scalarization,
            Convolution convolution,
            Broadcast broadcast,
            Operation operation
    ){
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

        _activation = activation;
        _scalarization = scalarization;
        _convolution = convolution;
        _broadcast = broadcast;
        _operation = operation;

        _REGISTER.add(this);
        _LOOKUP.put(identifier, this);
        if(
            identifier
                .replace((""+((char)171)), "")
                .replace((""+((char)187)), "")
                .matches("[a-z]")
        ){
            if(identifier.contains((""+((char)171)))) {
                _LOOKUP.put(identifier.replace((""+((char)171)), "<<"), this);
            }
            if(identifier.contains((""+((char)187)))) {
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

    @Override
    public String getName(){
        return _name;
    }

    //-----------------

    @Override
    public Activation getActivation(){
        return _activation;
    }

    @Override
    public boolean supportsActivation(){
        return _activation!=null;
    }

    //-----------------

    @Override
    public Scalarization getScalarization(){
        return _scalarization;
    }

    @Override
    public boolean supportsScalar(){
        return _scalarization != null;
    }

    //-----------------

    @Override
    public Convolution getConvolution(){
        return _convolution;
    }

    @Override
    public boolean supportsConvolution(){
        return _convolution!=null;
    }

    //-----------------

    @Override
    public Broadcast getBroadcast(){
        return _broadcast;
    }

    @Override
    public boolean supportsBroadcast(){
        return _broadcast!=null;
    }

    //-----------------

    @Override
    public Operation getOperation(){
        return _operation;
    }

    @Override
    public boolean supportsOperation(){
        return _operation!=null;
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

    public boolean allowsForward(Tsr[] inputs)
    {
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

    @Override
    public ADAgent getADAgentOf(Function f, Tsr[] inputs, int i, boolean forward)
    {
        Function MUL   = FunctionBuilder.build("(I[0]*I[1])", false);
        if(forward)
        {
            Tsr d = f.derive(inputs, i);
            return new ADAgent(
                    ()->d,
                    (t, derivative) -> MUL.activate(new Tsr[]{derivative, d}),
                    null
            );
        } else {
            if (this.isOperation() && !this.isConvection())
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
                Function INV_X = FunctionBuilder.build(
                        "I[0]" +
                                identifier() +
                                ">>I[1]" +
                                identifier() +
                                ">>I[2]",
                        false
                );
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
