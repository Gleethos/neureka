package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Multiplication extends OperationType {

    public Multiplication(){

        super(
                "multiply", "*", false, false, false, true, false,
                null,
                new Scalarization("output = input1 * value;\n",
                        "if(d==0){output = value;}else{output = input1;}\n",
                        (inputs, value, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * value;
                            } else {
                                if (d == 0) return (t0Idx, t1Idx, t2Idx) -> value;
                                else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                            }
                        }),
                null,
                new Broadcast("",
                        "",
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            double[] t2_val = inputs[2].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if (d == 0) return t2_val[inputs[2].i_of_idx(t2Idx)];
                                    else return t1_val[inputs[1].i_of_idx(t1Idx)];
                                };
                            }
                        }),
                    null
        );
        new OperationType(
                "", ((char) 171) + "*", false, false, false, false, false,
                null, null, null, null, null
        );
        new OperationType(
                "", "*" + ((char) 187), false, false, false, false, false,
                null, null, null, null, null
        );

        // Convolution:

        new OperationType(
                "convolve", "x", false, false, true, false, false,
                null,
                null,
                new Convolution(
                        "",
                        "",
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            double[] t2_val = inputs[2].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if (d == 0) return t2_val[inputs[2].i_of_idx(t2Idx)];
                                    else return t1_val[inputs[1].i_of_idx(t1Idx)];
                                };
                            }
                        }
                ),
                null,
                null
        );
        new OperationType(
                "", ((char) 171) + "x", false, false, true, false, false,
                null, null, null, null, null
        );
        new OperationType(
                "", "x" + ((char) 187), false, false, true, false, false,
                null, null, null, null, null
        );



    }



}
