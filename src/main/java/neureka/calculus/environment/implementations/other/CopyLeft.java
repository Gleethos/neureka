package neureka.calculus.environment.implementations.other;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.Activation;

public class CopyLeft extends OperationType {

    public CopyLeft(){

        super(
                "", "<", 2,true, false, false, false, false
        );
        set(Activation.class,
                new Activation("output = input;\n",
                        "output = input;\n",
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                            else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                        })
        );
    }



}
