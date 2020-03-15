package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;

public class ReLU extends OperationType {

    public ReLU(){
        super(
                "relu",
                "relu",
                false,
                false,
                false,
                true,
                true,

                new Activation(
                        "if (input >= 0) {  output = input; } else { output = input * (float)0.01; }\n",
                        "if (input >= 0) { output = (float)1; } else { output = (float)0.01; }\n",
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if(t1_val[inputs[1].i_of_idx(t1Idx)]>=0) return t1_val[inputs[1].i_of_idx(t1Idx)];
                                    else return t1_val[inputs[1].i_of_idx(t1Idx)]*0.01;
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if(t1_val[inputs[1].i_of_idx(t1Idx)]>=0) return 1;
                                    else return 0.01;
                                };
                            }
                        }),
                null,
                null,
                null,
                null
        );


    }


}
