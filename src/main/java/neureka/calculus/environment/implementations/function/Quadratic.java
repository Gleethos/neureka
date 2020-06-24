package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Quadratic extends OperationType {

    public Quadratic(){

        super(
                "quadratic",
                "quad",
                1,
                false,
                false,
                false,
                true,
                true
        );
        set(
                Activation.class,
                new Activation(
                        "output = input*input;\n",
                        "output = 2*input;\n",
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return input * input;
                                };
                            } else return (t0Idx, t1Idx, t2Idx) -> 2 * t1_val[inputs[1].i_of_idx(t1Idx)];
                        })
        );
    }


}
