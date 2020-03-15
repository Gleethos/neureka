package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Addition extends OperationType {

    private static final OperationCreator _creator =
            (inputs, d) -> {
                double[] t1_val = inputs[1].value64();
                double[] t2_val = inputs[2].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + t2_val[inputs[2].i_of_idx(t2Idx)];
                } else {
                    return (t0Idx, t1Idx, t2Idx) -> 1.0;
                }
            };

    private static final Broadcast _broadcast = new Broadcast(
            "value = src1 + src2;\n",
            "value += 1 * drain;\n",
            _creator
    );

    public Addition(){

        super(
                "add",
                "+",
                true, false, false, true, false,
                null,
                new Scalarization(
                        "output = input1 + value;\n",
                        "output = 1;\n",
                        (inputs, value, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + value;
                            else return (t0Idx, t1Idx, t2Idx) -> 1;
                        }),
                null,
                _broadcast,
                new Operation(
                        "output = input1 + input2;\n",
                        "output = 1;\n",
                        _creator
                )
        );
        new OperationType(
                "", ((char) 171) + "+", true, false, false, false, false,
                null,
                null,
                null,
                _broadcast,
                null
        );
        new OperationType(
                "", "+" + ((char) 187), true, false, false, false, false,
                null,
                null,
                null,
                _broadcast,
                null
        );

        // Convolutoion:

        new OperationType(
                "add", "a", true, false, true, false, false,
                null,
                null,
                new Convolution(
                        "value = src1 + src2;\n",
                        "value += 1 * drain;\n",
                        null
                ),
                null,//_broadcast,
                null
        );
        new OperationType(
                "", ((char) 171) + "a", true, false, true, false, false,
                null,
                null,
                null,
                null,//_broadcast,
                null
        );
        new OperationType(
                "", "a" + ((char) 187), true, false, true, false, false,
                null,
                null,
                null,
                null,//_broadcast,
                null
        );


    }

}
