package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;

public class Absolute extends OperationType {

    public Absolute(){

        super("absolute", "abs" , true, false, false, true, true,
                "output = fabs(input);\n",
                "output = (input < 0) ? -1 : 1;\n",
                (inputs, d)->{
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) {
                        return (t0Idx, t1Idx, t2Idx) -> Math.abs(t1_val[inputs[1].i_of_idx(t1Idx)]);
                    } else {
                        return (t0Idx, t1Idx, t2Idx) -> (t1_val[inputs[1].i_of_idx(t1Idx)] < 0) ? -1 : 1;
                    }
                },
                "",
                "",
                null,
                "",
                "",
                null
        );

    }

}
