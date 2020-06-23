package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;


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
                "divide", "/", -1, true, false, false, false, false
        );
        set(Scalarization.class,
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
                )
        );
        set(
                Broadcast.class,
                new Broadcast(
                        "value = src1 / src2;\n",
                        "if(d==0){\n" +
                                "    value += (1/handle) * drain;\n" +
                                "} else {\n" +
                                "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                                "}",
                        _creator
                )
        );
        set(Operation.class,
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
                "inv_division_left", ((char) 171) + "/", 3, true, false, false, false, false
        );
        new OperationType(
                "inv_division_right", "/" + ((char) 187), 3, true, false, false, false, false
        );

        // Convolution:

        new OperationType(
                "divide", "d", 2, true, false, true, false, false
        ).set(
                Convolution.class,
                new Convolution(
                        "value = src1 / src2;\n",
                        "if(d==0) {\n" +
                                "    value += (1/handle) * drain;\n" +
                                "} else {\n" +
                                "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                                "}",
                        null
                )
        );

        new OperationType(
                "", ((char) 171) + "d", 3, true, false, true, false, false
        );
        new OperationType(
                "", "d" + ((char) 187), 3, true, false, true, false, false
        );

    }



}
