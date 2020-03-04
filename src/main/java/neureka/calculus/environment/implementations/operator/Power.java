package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Power extends OperationType
{
    public Power(){

        super("power", "^", false, false, false, false, false, "", "", null,
                "output = pow(input1, value);",
                "if(d==0){\n" +
                        "    output = value * pow(input1, value-(float)1 );\n" +
                        "} else {\n" +
                        "    output = pow(input1, value) * log(value);\n" +
                        "}",
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
                "",
                "",
                null
        );

    }

}
