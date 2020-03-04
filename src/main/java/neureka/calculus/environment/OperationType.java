package neureka.calculus.environment;

import neureka.Tsr;
import neureka.acceleration.CPU;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.implementations.function.*;
import neureka.calculus.environment.implementations.indexer.Product;
import neureka.calculus.environment.implementations.indexer.Summation;
import neureka.calculus.environment.implementations.operator.Power;
import neureka.calculus.factory.assembly.FunctionBuilder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class OperationType implements Type
{
    private static Map<String, OperationType> _LOOKUP = new HashMap<>();

    private static ArrayList<OperationType> _REGISTER = new ArrayList<>();

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

    protected static OperationType[] _TYPES;

    private static int _ID = 0;

    protected int _id;
    protected String _name;
    protected String  _identifier;
    protected boolean _isFunction;
    protected boolean _isOperation;
    protected boolean _isIndexer;
    protected boolean _isConvection;
    protected boolean _isCommutative;
    protected boolean _isAssociative;

    protected String _operationAsString;
    protected String _deriviationAsString;
    protected OperationType.OperationCreator _operationCreator;

    protected String _scalarOperationAsString;
    protected String _scalarDeriviationAsString;
    protected OperationType.ScalarOperationCreator _scalarOperationCreator;

    protected String _broadcastOperationAsString;
    protected String _broadcastDeriviationAsString;
    protected OperationType.OperationCreator _broadcastOperationCreator;

    protected void _construct(
            String name,
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,
            OperationType.OperationCreator creator,
            String operationAsString,
            String deriviationAsString,
            OperationType.ScalarOperationCreator scalarOperationCreator,
            String scalarOperationAsString,
            String scalarDeriviationAsString,

            String broadcastOperationAsString,
            String broadcastDerivativeAsString,
            OperationType.OperationCreator broadcastCreator
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
        _operationCreator = creator;
        _operationAsString = operationAsString;
        _deriviationAsString = deriviationAsString;
        _scalarOperationCreator = scalarOperationCreator;
        _scalarOperationAsString = scalarOperationAsString;
        _scalarDeriviationAsString = scalarDeriviationAsString;

        _broadcastOperationAsString = broadcastOperationAsString;
        _broadcastDeriviationAsString =  broadcastDerivativeAsString;
        _broadcastOperationCreator = broadcastCreator;

        _REGISTER.add(this);
        _LOOKUP.put(identifier, this);
        if(identifier.equals((((char)171))+"x")) _LOOKUP.put("<<x", this);
        else if(identifier.equals("x"+((char)187))) _LOOKUP.put("x>>", this);
    }



    public static OperationType[] all(){
        return _TYPES;
    }


    public static int COUNT(){
        return _ID;
    }

    //==================================================================================================================

    public String getName(){
        return _name;
    }

    //-----------------

    public OperationCreator getCreator(){
        return _operationCreator;
    }

    public String getOperationAsString(){
        return _operationAsString;
    }

    public String getDeriviationAsString(){
        return _deriviationAsString;
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

    public OperationCreator getBroadcastOperationCreator() {
        return _broadcastOperationCreator;
    }

    public String getBroadcastOperationAsString(){
        return _broadcastOperationAsString;
    }

    public String getBroadcastDeriviationAsString(){
        return _broadcastDeriviationAsString;
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
        return !_scalarOperationAsString.equals("")
                &&
                !_scalarDeriviationAsString.equals("");
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





    static
    {
        _TYPES = new OperationType[]
        {
                new ReLU(),
                new Sigmoid(),
                new Tanh(),
                new Quadratic(),
                new Ligmoid(),
                new Identity(),
                new Gaussian(),
                new Absolute(),
                new Sinus(),
                new Cosinus(),

                // Indexer:
                new Summation(),
                new Product(),

                //-=-
                // Operations (auto broadcast):
                new OperationType("random", "rand", false, false, false, false, false, "", "", null,
                        "output = pow(input1, value);",
                        "if(d==0){\n" +
                                "    output = value * pow(input1, value-(float)1 );\n" +
                                "} else {\n" +
                                "    output = pow(input1, value) * log(value);\n" +
                                "}",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            return (t0Idx, t1Idx, t2Idx) ->
                                    Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value);

                        },
                        "",
                        "",
                        null
                ),

                //-=-
                // Operations (auto broadcast):
                new Power(),
                new OperationType("inv_power_left", ((char)171)+"^", false, false, false, false, false, "", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("inv_power_right", "^"+((char)187), false, false, false, false, false, "", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("divide", "/", false, false, false, false, false, "", "",null,
                        "output = input1 / value;\n",
                        "if(d==0){\n" +
                                "    output = 1/value;\n" +
                                "} else {\n" +
                                "    output = -value /(float)pow(input1, 2.0f);\n" +
                                "}",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] / value;
                            } else {
                                if(d==0) return (t0Idx, t1Idx, t2Idx) -> 1 / value;
                                else return (t0Idx, t1Idx, t2Idx) -> -value/Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], 2);
                            }
                        },
                        "",
                        "",
                        null
                ),
                new OperationType("inv_division_left", ((char)171)+"/", false, false, false, false, false, "", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("inv_division_right", "/"+((char)187), false, false, false, false, false, "", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("multiply", "*", false, false, false, true, false,  "", "", null,
                        "output = input1 * value;\n",
                        "if(d==0){output = value;}else{output = input1;}\n",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * value;
                            } else {
                                if(d==0) return (t0Idx, t1Idx, t2Idx) -> value;
                                else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                            }
                        },
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"*", false, false, false, false, false, "", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "*"+((char)187), false, false, false, false, false,  "", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("modulo", "%", false, false, false, false, false, "", "",null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"%", false, false, false, false, false,  "", "",null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "%"+((char)187), false, false, false, false, false, "", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("subtract", "-", false, false, false, false, false, "", "",null,
                        "output = input1 - value;\n",
                        "if(d==0){\n" +//drn and src2 switch:
                                "    output = 1;\n" +
                                "} else {\n" +
                                "    output = -1;" +
                                "}",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] - value;
                            } else {
                                if(d==0) return (t0Idx, t1Idx, t2Idx) -> 1;
                                else return (t0Idx, t1Idx, t2Idx) -> -1;
                            }
                        },
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"-", false, false, false, false, false, "", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "-"+((char)187), false, false, false, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("add", "+", false, false, false, true, false, "", "",null,
                        "output = input1 + value;\n",
                        "output = 1;\n",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + value;
                            else return (t0Idx, t1Idx, t2Idx) -> 1;
                        },
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"+", false, false, false, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "+"+((char)187), false, false, false, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                // Convolution:
                new OperationType("convolve", "x", false, false, true, false, false, "", "",null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"x", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "x"+((char)187), false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("", "d", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"d", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "d"+((char)187), false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("", "p", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"p", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "p"+((char)187), false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("", "a", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"a", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "a"+((char)187), false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                new OperationType("", "s", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", ((char)171)+"s", false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", "s"+((char)187), false, false, true, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                // (char)171 -> <<    // (char)187 -> >>
                //---
                // Reshape:
                new OperationType("", ",", false, false, false, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                //---
                // Injecting:
                new OperationType("", "<", false, false, false, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                ),
                new OperationType("", ">", false, false, false, false, false,"", "", null,
                        "",
                        "",
                        null,
                        "",
                        "",
                        null
                )
                //---
        };
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

            String operationAsString,
            String deriviationAsString,
            OperationCreator creator,

            String scalarOperationAsString,
            String scalarDeriviationAsString,
            ScalarOperationCreator scalarOperationCreator,

            String broadcastOperationAsString,
            String broadcastDerivativeAsString,
            OperationCreator broadcastCreator
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
                creator,
                operationAsString,
                deriviationAsString,
                scalarOperationCreator,
                scalarOperationAsString,
                scalarDeriviationAsString,

                broadcastOperationAsString,
                broadcastDerivativeAsString,
                broadcastCreator
        );
    }

    protected OperationType(
            String name,
            String  identifier,
            boolean isFunction,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,

            String operationAsString,
            String deriviationAsString,
            OperationCreator creator,

            String scalarOperationAsString,
            String scalarDeriviationAsString,
            ScalarOperationCreator scalarOperationCreator,

            String broadcastOperationAsString,
            String broadcastDerivativeAsString,
            OperationCreator broadcastCreator
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
                creator,
                operationAsString,
                deriviationAsString,
                scalarOperationCreator,
                scalarOperationAsString,
                scalarDeriviationAsString,

                broadcastOperationAsString,
                broadcastDerivativeAsString,
                broadcastCreator
        );
    }




}
