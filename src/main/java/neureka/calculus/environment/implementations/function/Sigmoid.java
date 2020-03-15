package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;

public class Sigmoid extends OperationType {

    public Sigmoid(){

        super(
                "sigmoid",
                "sig" ,
                false,
                false,
                false,
                true,
                true,
                new Activation(
                        "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                        "output = input * (1 - input);\n",
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> 1 / (1 + Math.pow(Math.E, -t1_val[inputs[1].i_of_idx(t1Idx)]));
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                                    return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
                                };
                            }
                        }
                ),
                null,
                null,
                null,
                null
        );

    }



}




