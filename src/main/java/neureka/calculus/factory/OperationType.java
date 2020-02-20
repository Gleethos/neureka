package neureka.calculus.factory;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class OperationType {

    private static Map<String, OperationType> _LOOKUP = new HashMap<>();

    private static ArrayList<OperationType> _REGISTER = new ArrayList<>();

    private static int _ID = 0;

    public static OperationType LOOKUP(String identifier){
        return _LOOKUP.getOrDefault(identifier, null);
    }

    public static OperationType REGISTER(int index){
        return _REGISTER.get(index);
    }

    public static int COUNT(){
        return _ID;
    }

    private int _id;
    private String  _identifier;
    private boolean _isFunction;
    private boolean _isOperation;
    private boolean _isIndexer;
    private boolean _isConvection;
    private boolean _isCommutative;
    private boolean _isAssociative;

    private static OperationType[] _TYPES;

    static  {
        _TYPES = new OperationType[]{
                new OperationType("relu", true, false, false, true, true),
                new OperationType("sig" , true, false, false, true, true),
                new OperationType("tanh", true, false, false, true, true),
                new OperationType("quad", true, false, false, true, true),
                new OperationType("lig" , true, false, false, true, true),
                new OperationType("idy" , true, false, false, true, true),
                new OperationType("gaus", true, false, false, true, true),
                new OperationType("abs" , true, false, false, true, true),
                new OperationType("sin" , true, false, false, true, true),
                new OperationType("cos" , true, false, false, true, true),

                // Indexer:
                new OperationType("sum" , false, false, true, false, true, true),
                new OperationType("prod", false, false,  true, false, true, true),

                // Operations (auto broadcast):
                new OperationType("^", false, false, false, false, false),
                new OperationType(((char)171)+"^", false, false, false, false, false),
                new OperationType("^"+((char)187), false, false, false, false, false),

                new OperationType("/", false, false, false, false, false),
                new OperationType(((char)171)+"/", false, false, false, false, false),
                new OperationType("/"+((char)187), false, false, false, false, false),

                new OperationType("*", false, false, false, true, false),
                new OperationType(((char)171)+"*", false, false, false, false, false),
                new OperationType("*"+((char)187), false, false, false, false, false),

                new OperationType("%", false, false, false, false, false),
                new OperationType(((char)171)+"%", false, false, false, false, false),
                new OperationType("%"+((char)187), false, false, false, false, false),

                new OperationType("-", false, false, false, false, false),
                new OperationType(((char)171)+"-", false, false, false, false, false),
                new OperationType("-"+((char)187), false, false, false, false, false),

                new OperationType("+", false, false, false, true, false),
                new OperationType(((char)171)+"+", false, false, false, false, false),
                new OperationType("+"+((char)187), false, false, false, false, false),

                // Convolution:
                new OperationType("x", false, false, true, false, false),
                new OperationType(((char)171)+"x", false, false, true, false, false),
                new OperationType("x"+((char)187), false, false, true, false, false),

                new OperationType("d", false, false, true, false, false),
                new OperationType(((char)171)+"d", false, false, true, false, false),
                new OperationType("d"+((char)187), false, false, true, false, false),

                new OperationType("p", false, false, true, false, false),
                new OperationType(((char)171)+"p", false, false, true, false, false),
                new OperationType("p"+((char)187), false, false, true, false, false),

                new OperationType("a", false, false, true, false, false),
                new OperationType(((char)171)+"a", false, false, true, false, false),
                new OperationType("a"+((char)187), false, false, true, false, false),

                new OperationType("s", false, false, true, false, false),
                new OperationType(((char)171)+"s", false, false, true, false, false),
                new OperationType("s"+((char)187), false, false, true, false, false),
                // (char)171 -> <<    // (char)187 -> >>

                // Reshape:
                new OperationType(",", false, false, false, false, false),

                // Injecting:
                new OperationType("<", false, false, false, false, false),
                new OperationType(">", false, false, false, false, false),
        };
    }


    public OperationType(
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative
    ) {
        _construct(
                identifier,
                isFunction,
                isOperation,
                isIndexer,
                isConvection,
                isCommutative,
                isAssociative
        );
    }

    public OperationType(
            String  identifier,
            boolean isFunction,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative
    ) {
        _construct(
                identifier,
                isFunction,
                !isFunction,
                isIndexer,
                isConvection,
                isCommutative,
                isAssociative
        );
    }

    private void _construct(
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative
    ) {
        _id = _ID;
        _ID++;
        _identifier = identifier;
        _isFunction = isFunction;
        _isOperation = isOperation;
        _isIndexer = isIndexer;
        _isConvection = isConvection;
        _isCommutative = isCommutative;
        _isAssociative = isAssociative;

        _REGISTER.add(this);
        _LOOKUP.put(identifier, this);
        if(identifier.equals((((char)171))+"x")) _LOOKUP.put("<<x", this);
        else if(identifier.equals("x"+((char)187))) _LOOKUP.put("x>>", this);
    }

    public static OperationType[] all(){
        return _TYPES;
    }


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


    private static Function MUL = FunctionBuilder.build("(I[0]*I[1])", false);
    private static Function ADD = FunctionBuilder.build("(I[0]+I[1])", false);
    private static Function INV_X = FunctionBuilder.build("I[0]x>>I[1]x>>I[2]", false);

    public boolean allowsForward(Tsr[] inputs){
        if(this.isConvection()) return false;
        if(this.identifier().equals(",")) return false; //Reshape
        Tsr last = null;
        for(Tsr t : inputs){
            if(last!=null){
                if(!last.shape().equals(t.shape())){
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
