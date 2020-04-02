package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Division extends OperationType {

    private static final OperatorCreator _creator =
    (inputs, d) -> {
        double[] t1_val = inputs[1].value64();
        double[] t2_val = inputs[2].value64();
        if (d < 0) {
            return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] / t2_val[inputs[2].i_of_idx(t2Idx)];
        } else {
            return (t0Idx, t1Idx, t2Idx) -> {
                if (d == 0) {
                    return 1 / t2_val[inputs[2].i_of_idx(t2Idx)];
                } else {
                    return -(t1_val[inputs[1].i_of_idx(t1Idx)] / Math.pow(t2_val[inputs[2].i_of_idx(t2Idx)], 2));
                }
            };
        }
    };

    public Division() {

        super(
                "divide", "/", true, false, false, false, false,
                null,
                new Scalarization(
                        "output = input1 / value;\n",
                        "if(d==0){\n" +
                                "    output = 1/value;\n" +
                                "} else {\n" +
                                "    output = -value /(float)pow(input1, 2.0f);\n" +
                                "}",
                        (inputs, value, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] / value;
                            } else {
                                if (d == 0) return (t0Idx, t1Idx, t2Idx) -> 1 / value;
                                else return (t0Idx, t1Idx, t2Idx) -> -value / Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], 2);
                            }
                        }
                ),
                null,
                new Broadcast(
                        "value = src1 / src2;\n",
                        "if(d==0){\n" +
                                   "    value += (1/handle) * drain;\n" +
                                   "} else {\n" +
                                   "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                                   "}",
                            _creator
                        ),
                new Operation(
                        "output = input1 / input2;\n",
                        "if(d==0){\n" +
                                "    output = 1/input2;\n" +
                                "} else {\n" +
                                "    output = -input2 /(float)pow(input1, 2.0f);\n" +
                                "}",
                        _creator
                )

        );
        new OperationType(
                "inv_division_left", ((char) 171) + "/", true, false, false, false, false,
                null, null, null, null, null
        );
        new OperationType(
                "inv_division_right", "/" + ((char) 187), true, false, false, false, false,
                null, null, null, null, null
        );

        // Convolution:

        new OperationType(
                "divide", "d", true, false, true, false, false,
                null,
                null,
                new Convolution(
                        "value = src1 / src2;\n",
                        "if(d==0) {\n" +
                                   "    value += (1/handle) * drain;\n" +
                                   "} else {\n" +
                                   "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                                   "}",
                        null
                ),
                null,
                null
        );
        new OperationType(
                "", ((char) 171) + "d", true, false, true, false, false,
                null, null, null, null, null
        );
        new OperationType(
                "", "d" + ((char) 187), true, false, true, false, false,
                null, null, null, null, null
        );

    }



}
