package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Power extends OperationType
{
    public Power(){

        super("power", "^", false, false, false, false, false,
                null,
                new Scalarization("output = pow(input1, value);",
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
                        }),
                null,
                new Broadcast("",
                        "",
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            double[] t2_val = inputs[2].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], t2_val[inputs[2].i_of_idx(t2Idx)]);
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if (d == 0) {
                                        return t2_val[inputs[2].i_of_idx(t2Idx)]
                                                * Math.pow(
                                                t1_val[inputs[1].i_of_idx(t1Idx)],
                                                t2_val[inputs[2].i_of_idx(t2Idx)] - 1
                                        );
                                    } else {
                                        return Math.pow(
                                                t1_val[inputs[1].i_of_idx(t1Idx)],
                                                t2_val[inputs[2].i_of_idx(t2Idx)]
                                        ) * Math.log(t1_val[inputs[1].i_of_idx(t1Idx)]);
                                    }
                                };
                            }
                        })

        );


        new OperationType(
                "inv_power_left", ((char)171)+"^", false, false, false, false, false,
                null, null, null, null
        );


        new OperationType("inv_power_right", "^" + ((char) 187), false, false, false, false, false,
                null, null, null, null
        );

        // Convolution:


        new OperationType(
                "", "p", false, false, true, false, false,
                null, null, null, null
        );
        new OperationType(
                "", ((char) 171) + "p", false, false, true, false, false,
                null, null, null, null
        );
        new OperationType(
                "", "p" + ((char) 187), false, false, true, false, false,
                null, null, null, null
        );




    }

}
