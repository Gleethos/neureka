package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Division extends OperationType {

    public Division() {

        super(
                "divide", "/", false, false, false, false, false,
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
                        "",
                        "",
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
                        })

        );
        new OperationType(
                "inv_division_left", ((char) 171) + "/", false, false, false, false, false,
                null, null, null, null
        );
        new OperationType(
                "inv_division_right", "/" + ((char) 187), false, false, false, false, false,
                null, null, null, null
        );

        // Convolution:

        new OperationType(
                "", "d", false, false, true, false, false,
                null, null, null, null
        );
        new OperationType(
                "", ((char) 171) + "d", false, false, true, false, false,
                null, null, null, null
        );
        new OperationType(
                "", "d" + ((char) 187), false, false, true, false, false,
                null, null, null, null
        );

    }



}
