package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;

public class Quadratic extends OperationType {

    public Quadratic(){

        super(
                "quadratic",
                "quad",
                true,
                false,
                false,
                true,
                true,
                new Activation(
                        "output = input*input;\n",
                        "output = 2*input;\n",
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return input * input;
                                };
                            } else return (t0Idx, t1Idx, t2Idx) -> 2 * t1_val[inputs[1].i_of_idx(t1Idx)];
                        }),
                null,
                null,
                null
        );

    }


}
