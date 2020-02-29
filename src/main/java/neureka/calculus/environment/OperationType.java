package neureka.calculus.environment;

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
    public interface ScalarOperationCreator {
        CPU.exec.Operator create(Tsr[] inputs, double scalar, int d);
    }
    
    public static OperationType instance(String identifier){
        return _LOOKUP.get(identifier);
    }

    public static int COUNT(){
        return _ID;
    }

    private int _id;
    private String _name;
    private String  _identifier;
    private boolean _isFunction;
    private boolean _isOperation;
    private boolean _isIndexer;
    private boolean _isConvection;
    private boolean _isCommutative;
    private boolean _isAssociative;
    private OperationCreator _operationCreator;
    private String _operationAsString;
    private String _deriviationAsString;
    private ScalarOperationCreator _scalarOperationCreator;
    private String _scalarOperationAsString;
    private String _scalarDeriviationAsString;

    private static OperationType[] _TYPES;

    static  {

        _TYPES = new OperationType[] {
                new OperationType("relu", "relu", true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if(t1_val[inputs[1].i_of_idx(t1Idx)]>=0) return t1_val[inputs[1].i_of_idx(t1Idx)];
                                    else return t1_val[inputs[1].i_of_idx(t1Idx)]*0.01;
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if(t1_val[inputs[1].i_of_idx(t1Idx)]>=0) return 1;
                                    else return 0.01;
                                };
                            }
                        },
                        "if (input >= 0) {  output = input; } else { output = input * (float)0.01; }\n",
                        "if (input >= 0) { output = (float)1; } else { output = (float)0.01; }\n",
                        null,
                        "",
                        ""
                ),
                new OperationType("sigmoid", "sig" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> 1 / (1 + Math.pow(Math.E, -t1_val[inputs[1].i_of_idx(t1Idx)]));
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
                                };
                            }
                        },
                        "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                        "output = input * (1 - input);\n",
                        null,
                        "",
                        ""
                ),
                new OperationType("tanh", "tanh", true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return input / Math.pow(1 + Math.pow(input, 2), 0.5);
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return 1 - Math.pow(input / Math.pow(1 + Math.pow(input, 2), 0.5), 2);
                                };
                            }
                        },
                        "output = input/pow(1+pow(input, 2.0f), 0.5f);\n",
                        "output = 1-pow(input/pow((1.0f+pow(input,2.0f)),0.5f), 2.0f);\n",
                        null,
                        "",
                        ""
                ),
                new OperationType("quadratic", "quad", true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return input * input;
                                };
                            } else return (t0Idx, t1Idx, t2Idx) -> 2 * t1_val[inputs[1].i_of_idx(t1Idx)];
                        },
                        "output = input*input;\n",
                        "output = 2*input;\n",
                        null,
                        "",
                        ""
                ),
                new OperationType("ligmoid", "lig" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.log(1 + Math.pow(Math.E, t1_val[inputs[1].i_of_idx(t1Idx)]));
                            else return (t0Idx, t1Idx, t2Idx) -> 1 / (1 + Math.pow(Math.E, -t1_val[inputs[1].i_of_idx(t1Idx)]));
                        },
                        "output = \n" +
                                "(\n" +
                                "        (float) log(\n" +
                                "            1+pow(\n" +
                                "                (float)\n" +
                                "                M_E,\n" +
                                "                (float)\n" +
                                "                input\n" +
                                "            )\n" +
                                "        )\n" +
                                "    );",
                        "output =\n" +
                                "    1 /\n" +
                                "        (1 + (float) pow(\n" +
                                "                (float)M_E,\n" +
                                "                (float)input\n" +
                                "            )\n" +
                                "        );\n",
                        null,
                        "",
                        ""
                ),
                new OperationType("identity", "idy" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                            else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                        },
                        "output = input;\n",
                        "output = input;\n",
                        null,
                        "output = value;\n",
                        "output = value;\n"
                ),
                new OperationType("gaussian", "gaus", true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> Math.pow(Math.E, -Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], 2));
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return -2 * input * Math.pow(Math.E, -Math.pow(input, 2));
                                };

                            }
                        },
                        "output =\n" +
                                "    (float)pow(\n" +
                                "        (float)M_E,\n" +
                                "        -(float)pow(\n" +
                                "            (float)input,\n" +
                                "            (float)2\n" +
                                "        )\n" +
                                "    );\n",
                        "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                        null,
                        "",
                        ""
                ),
                new OperationType("absolute", "abs" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> Math.abs(t1_val[inputs[1].i_of_idx(t1Idx)]);
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> (t1_val[inputs[1].i_of_idx(t1Idx)] < 0) ? -1 : 1;
                            }
                        },
                        "output = fabs(input);\n",
                        "output = (input < 0) ? -1 : 1;\n",
                        null,
                        "",
                        ""
                ),
                new OperationType("sinus", "sin" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.sin(t1_val[inputs[1].i_of_idx(t1Idx)]);
                            else return (t0Idx, t1Idx, t2Idx) -> Math.cos(t1_val[inputs[1].i_of_idx(t1Idx)]);
                        },
                        "output = sin(input);\n",
                        "output = cos(input);\n",
                        null,
                        "",
                        ""
                ),
                new OperationType("cosinus", "cos" , true, false, false, true, true,
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.cos(t1_val[inputs[1].i_of_idx(t1Idx)]);
                            else return (t0Idx, t1Idx, t2Idx) -> -Math.sin(t1_val[inputs[1].i_of_idx(t1Idx)]);
                        },
                        "output = cos(input);\n",
                        "output = -sin(input);\n",
                        null,
                        "",
                        ""
                ),

                // Indexer:
                new OperationType("summation", "sum" , false, false, true, false, true, true,
                        null,
                        "output = input;",
                        "output = 1;",
                        null,
                        "",
                        ""
                ),
                new OperationType("product", "prod", false, false,  true, false, true, true,
                        null,
                        "output = input;",
                        "output = 1;",
                        null,
                        "",
                        ""
                ),
                //-=-
                // Operations (auto broadcast):
                new OperationType("power", "^", false, false, false, false, false, null, "", "", 
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value);
                            } else {
                                if(d==0){
                                    return (t0Idx, t1Idx, t2Idx) -> value*Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value-1);
                                } else {
                                    return (t0Idx, t1Idx, t2Idx) -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value)*Math.log(value);
                                }
                            }
                        },
                        "output = pow(input1, value);",
                        "if(d==0){\n" +
                                "    output = value * pow(input1, value-(float)1 );\n" +
                                "} else {\n" +
                                "    output = pow(input1, value) * log(value);\n" +
                                "}"
                ),
                new OperationType("inv_power_left", ((char)171)+"^", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("inv_power_right", "^"+((char)187), false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("divide", "/", false, false, false, false, false, null, "", "",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] / value;
                            } else {
                                if(d==0) return (t0Idx, t1Idx, t2Idx) -> 1 / value;
                                else return (t0Idx, t1Idx, t2Idx) -> -value/Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], 2);
                            }
                        },
                        "output = input1 / value;\n",
                        "if(d==0){\n" +
                                "    output = 1/value;\n" +
                                "} else {\n" +
                                "    output = -value /(float)pow(input1, 2.0f);\n" +
                                "}"
                ),
                new OperationType("inv_division_left", ((char)171)+"/", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("inv_division_right", "/"+((char)187), false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("multiply", "*", false, false, false, true, false, null, "", "",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * value;
                            } else {
                                if(d==0) return (t0Idx, t1Idx, t2Idx) -> value;
                                else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                            }
                        },
                        "output = input1 * value;\n",
                        "if(d==0){output = value;}else{output = input1;}\n"
                ),
                new OperationType("", ((char)171)+"*", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "*"+((char)187), false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("modulo", "%", false, false, false, false, false, null, "", "",
                        null,
                        "",
                        ""
                ),
                new OperationType("", ((char)171)+"%", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "%"+((char)187), false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("subtract", "-", false, false, false, false, false, null, "", "",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] - value;
                            } else {
                                if(d==0) return (t0Idx, t1Idx, t2Idx) -> 1;
                                else return (t0Idx, t1Idx, t2Idx) -> -1;
                            }
                        },
                        "output = input1 - value;\n",
                        "if(d==0){\n" +//drn and src2 switch:
                                "    output = 1;\n" +
                                "} else {\n" +
                                "    output = -1;" +
                                "}"
                ),
                new OperationType("", ((char)171)+"-", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "-"+((char)187), false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("add", "+", false, false, false, true, false, null, "", "",
                        (inputs, value, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + value;
                            else return (t0Idx, t1Idx, t2Idx) -> 1;
                        },
                        "output = input1 + value;\n",
                        "output = 1;\n"
                ),
                new OperationType("", ((char)171)+"+", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "+"+((char)187), false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                // Convolution:
                new OperationType("convolve", "x", false, false, true, false, false, null, "", "",
                        null,
                        "",
                        ""
                ),
                new OperationType("", ((char)171)+"x", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "x"+((char)187), false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("", "d", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", ((char)171)+"d", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "d"+((char)187), false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("", "p", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", ((char)171)+"p", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "p"+((char)187), false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("", "a", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", ((char)171)+"a", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "a"+((char)187), false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                new OperationType("", "s", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", ((char)171)+"s", false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", "s"+((char)187), false, false, true, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                // (char)171 -> <<    // (char)187 -> >>
                //---
                // Reshape:
                new OperationType("", ",", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                //---
                // Injecting:
                new OperationType("", "<", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                ),
                new OperationType("", ">", false, false, false, false, false, null, "", "", 
                        null,
                        "",
                        ""
                )
                //---
        };
    }


    public OperationType(
            String name,
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,
            OperationCreator creator,
            String operationAsString,
            String deriviationAsString,
            ScalarOperationCreator scalarOperationCreator,
            String scalarOperationAsString,
            String scalarDeriviationAsString
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
                scalarDeriviationAsString
        );
    }

    public OperationType(
            String name, 
            String  identifier,
            boolean isFunction,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,
            OperationCreator creator,
            String operationAsString,
            String deriviationAsString,
            ScalarOperationCreator scalarOperationCreator,
            String scalarOperationAsString,
            String scalarDeriviationAsString
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
                scalarDeriviationAsString
        );
    }

    private void _construct(
            String name,
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative,
            OperationCreator creator,
            String operationAsString,
            String deriviationAsString,
            ScalarOperationCreator scalarOperationCreator,
            String scalarOperationAsString,
            String scalarDeriviationAsString
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

        _REGISTER.add(this);
        _LOOKUP.put(identifier, this);
        if(identifier.equals((((char)171))+"x")) _LOOKUP.put("<<x", this);
        else if(identifier.equals("x"+((char)187))) _LOOKUP.put("x>>", this);
    }

    public static OperationType[] all(){
        return _TYPES;
    }

    public String getName(){
        return _name;
    }

    public OperationCreator getCreator(){
        return _operationCreator;
    }

    public String getOperationAsString(){
        return _operationAsString;
    }

    public String getDeriviationAsString(){
        return _deriviationAsString;
    }

    public ScalarOperationCreator getScalarOperationCreator(){
        return _scalarOperationCreator;
    }
    
    public String getScalarOperationAsString(){
        return _scalarOperationAsString;
    }
    
    public String getScalarDeriviationAsString(){
        return _scalarDeriviationAsString;
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

    public boolean supportsScalar(){
        return !_scalarOperationAsString.equals("")
                &&
               !_scalarDeriviationAsString.equals("");
    }

    //private static Function MUL    ;
    //private static Function ADD    ;
    //private static Function INV_X ;

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
