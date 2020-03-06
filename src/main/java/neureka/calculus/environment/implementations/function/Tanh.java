package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;

public class Tanh extends OperationType {

    public Tanh(){
        super("tanh", "tanh", true, false, false, true, true,
                new Activation(
                        "output = input/pow(1+pow(input, 2.0f), 0.5f);\n",
                        "output = 1-pow(input/pow((1.0f+pow(input,2.0f)),0.5f), 2.0f);\n",
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return input / Math.pow(1 + Math.pow(input, 2), 0.5);
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return 1 - Math.pow(input / Math.pow(1 + Math.pow(input, 2), 0.5), 2);
                                };
                            }
                        }),
                null,
                null,
                null
        );
    }





}

