package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Addition extends OperationType {

    private static final OperationCreator _creator = (inputs, d) -> {
        double[] t1_val = inputs[1].value64();
        double[] t2_val = inputs[2].value64();
        if (d < 0) {
            return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + t2_val[inputs[2].i_of_idx(t2Idx)];
        } else {
            return (t0Idx, t1Idx, t2Idx) -> 1.0;
        }
    };

    public Addition(){

        super(
                "add", "+", false, false, false, true, false, "", "", null,
                "output = input1 + value;\n",
                "output = 1;\n",
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + value;
                    else return (t0Idx, t1Idx, t2Idx) -> 1;
                },
                "",
                "",
                null,
                "",
                "",
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if (d < 0) {
                        return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + t2_val[inputs[2].i_of_idx(t2Idx)];
                    } else {
                        return (t0Idx, t1Idx, t2Idx) -> 1.0;
                    }
                }
        );
        new OperationType(
                "", ((char) 171) + "+", false, false, false, false, false,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                _creator
        );
        new OperationType(
                "", "+" + ((char) 187), false, false, false, false, false,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                _creator
        );

        // Convolutoion:


        new OperationType(
                "", "a", false, false, true, false, false,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                _creator
        );
        new OperationType(
                "", ((char) 171) + "a", false, false, true, false, false,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                null
        );
        new OperationType(
                "", "a" + ((char) 187), false, false, true, false, false,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                null,
                "",
                "",
                null
        );


    }

}
