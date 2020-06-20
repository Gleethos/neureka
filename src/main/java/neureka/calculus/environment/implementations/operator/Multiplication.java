package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Multiplication extends OperationType {


    private static final OperatorCreator _creator =
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
            };

    public Multiplication()
    {
        super(
                "multiply", "*", true, false, false, true, false
        );
        set(Scalarization.class,new Scalarization(
                "output = input1 * value;\n",
                "if(d==0){output = value;}else{output = input1;}\n",
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) {
                        return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * value;
                    } else {
                        if (d == 0) return (t0Idx, t1Idx, t2Idx) -> value;
                        else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                    }
                })
        ).set(Broadcast.class,
                new Broadcast(
                        "value = src1 * src2;\n",
                        "value += handle * drain;\n",
                        _creator
                )
        ).set(Operation.class,
                new Operation(
                "output = input1 * input2;\n",
                "if(d==0){output = input2;}else{output = input1;}\n",
                _creator
        ));
        OperatorCreator creator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                };
        new OperationType(
                "", ((char) 171) + "*", true, false, false, false, false
        ).set(Broadcast.class, new Broadcast(
                "value = src1 * src2;\n",
                "value += handle * drain;\n",
                creator
        ));
        new OperationType(
                "", "*" + ((char) 187), true, false, false, false, false
        ).set(Broadcast.class, new Broadcast(
                "value = src1 * src2;\n",
                "value += handle * drain;\n",
                creator
        ));

        // Convolution:

        Convolution convolution =
                new Convolution(
                        "value = src1 * src2;\n",
                        "value += handle * drain;\n",
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
                );

        new OperationType(
                "multiply", "x", true, false, true, false, false
        ).set(Convolution.class, convolution);
        new OperationType(
                "inv_convolve_mul_left", ((char) 171) + "x", true, false, true, false, false
        ).set(Convolution.class, convolution);
        new OperationType(
                "inv_convolve_mul_right", "x" + ((char) 187), true, false, true, false, false
        ).set(Convolution.class, convolution);



    }





}
