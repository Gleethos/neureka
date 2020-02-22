package neureka.calculus.factory;

import neureka.Tsr;
import neureka.acceleration.CPU;
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

    public static OperationType instance(int index){
        return _REGISTER.get(index);
    }
    
    public interface OperationCreator{
        CPU.exec.Operator create(Tsr[] inputs, int d);
    }
    
    public static OperationType instance(String identifier){
        return _LOOKUP.get(identifier);
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
    private OperationCreator _operationCreator;

    private static OperationType[] _TYPES;

    static  {
        _TYPES = new OperationType[]{
                new OperationType("relu", true, false, false, true, true, 
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if(t1_val[inputs[1].i_of_idx(t1Idx)]>=0){
                                        return t1_val[inputs[1].i_of_idx(t1Idx)];
                                    }
                                    return t1_val[inputs[1].i_of_idx(t1Idx)]*0.01;
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if(t1_val[inputs[1].i_of_idx(t1Idx)]>=0){
                                        return 1;
                                    }
                                    return 0.01;
                                };
                            }
                        }
                ),
                new OperationType("sig" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) ->
                                        1 / (1 + Math.pow(Math.E, -t1_val[inputs[1].i_of_idx(t1Idx)]));
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
                                };
                            }
                        }
                ),
                new OperationType("tanh", true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return ((input)) / Math.pow((1 + Math.pow(((input)), 2)), 0.5);
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
                                };
                            }
                        }
                ),
                new OperationType("quad", true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return ((input) * (input));
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) ->
                                        2 * t1_val[inputs[1].i_of_idx(t1Idx)];
                            }
                        }
                ),
                new OperationType("lig" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return (Math.log(1 + Math.pow(Math.E, input)));
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> 1 / (1 + Math.pow(Math.E, -t1_val[inputs[1].i_of_idx(t1Idx)]));
                            }
                        }
                ),
                new OperationType("idy" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                            }
                        }
                ),
                new OperationType("gaus", true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return Math.pow(Math.E, -Math.pow((input), 2));
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return -2 * ((input)) * Math.pow(Math.E, -Math.pow((input), 2));
                                };

                            }
                        }
                ),
                new OperationType("abs" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return Math.abs(input);
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return (input < 0) ? -1 : 1;
                                };
                            }
                        }
                ),
                new OperationType("sin" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return Math.sin(input);
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return Math.cos(input);
                                };
                            }
                        }
                ),
                new OperationType("cos" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return Math.cos(input);
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return -Math.sin(input);
                                };

                            }
                        }
                ),

                // Indexer:
                new OperationType("sum" , false, false, true, false, true, true, null),
                new OperationType("prod", false, false,  true, false, true, true, null),

                // Operations (auto broadcast):
                new OperationType("^", false, false, false, false, false, null),
                new OperationType(((char)171)+"^", false, false, false, false, false, null),
                new OperationType("^"+((char)187), false, false, false, false, false, null),

                new OperationType("/", false, false, false, false, false, null),
                new OperationType(((char)171)+"/", false, false, false, false, false, null),
                new OperationType("/"+((char)187), false, false, false, false, false, null),

                new OperationType("*", false, false, false, true, false, null),
                new OperationType(((char)171)+"*", false, false, false, false, false, null),
                new OperationType("*"+((char)187), false, false, false, false, false, null),

                new OperationType("%", false, false, false, false, false, null),
                new OperationType(((char)171)+"%", false, false, false, false, false, null),
                new OperationType("%"+((char)187), false, false, false, false, false, null),

                new OperationType("-", false, false, false, false, false, null),
                new OperationType(((char)171)+"-", false, false, false, false, false, null),
                new OperationType("-"+((char)187), false, false, false, false, false, null),

                new OperationType("+", false, false, false, true, false, null),
                new OperationType(((char)171)+"+", false, false, false, false, false, null),
                new OperationType("+"+((char)187), false, false, false, false, false, null),

                // Convolution:
                new OperationType("x", false, false, true, false, false, null),
                new OperationType(((char)171)+"x", false, false, true, false, false, null),
                new OperationType("x"+((char)187), false, false, true, false, false, null),

                new OperationType("d", false, false, true, false, false, null),
                new OperationType(((char)171)+"d", false, false, true, false, false, null),
                new OperationType("d"+((char)187), false, false, true, false, false, null),

                new OperationType("p", false, false, true, false, false, null),
                new OperationType(((char)171)+"p", false, false, true, false, false, null),
                new OperationType("p"+((char)187), false, false, true, false, false, null),

                new OperationType("a", false, false, true, false, false, null),
                new OperationType(((char)171)+"a", false, false, true, false, false, null),
                new OperationType("a"+((char)187), false, false, true, false, false, null),

                new OperationType("s", false, false, true, false, false, null),
                new OperationType(((char)171)+"s", false, false, true, false, false, null),
                new OperationType("s"+((char)187), false, false, true, false, false, null),
                // (char)171 -> <<    // (char)187 -> >>

                // Reshape:
                new OperationType(",", false, false, false, false, false, null),

                // Injecting:
                new OperationType("<", false, false, false, false, false, null),
                new OperationType(">", false, false, false, false, false, null),
        };
    }


    public OperationType(
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,
            OperationCreator creator
    ) {
        _construct(
                identifier,
                isFunction,
                isOperation,
                isIndexer,
                isConvection,
                isCommutative,
                isAssociative,
                creator
        );
    }

    public OperationType(
            String  identifier,
            boolean isFunction,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,
            OperationCreator creator
    ) {
        _construct(
                identifier,
                isFunction,
                !isFunction,
                isIndexer,
                isConvection,
                isCommutative,
                isAssociative,
                creator
        );
    }

    private void _construct(
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,
            OperationCreator creator
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
        _operationCreator = creator;
        
        _REGISTER.add(this);
        _LOOKUP.put(identifier, this);
        if(identifier.equals((((char)171))+"x")) _LOOKUP.put("<<x", this);
        else if(identifier.equals("x"+((char)187))) _LOOKUP.put("x>>", this);
    }

    public static OperationType[] all(){
        return _TYPES;
    }

    public OperationCreator getCreator(){
        return _operationCreator;
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
                    (t, derivative) ->{
                        return MUL.activate(new Tsr[]{derivative, d});
                    },
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
