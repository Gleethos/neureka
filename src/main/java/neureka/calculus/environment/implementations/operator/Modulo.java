package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Modulo extends OperationType {

    public Modulo(){

        super(
                "modulo", "%", -1, true, false, false, false, false
        );
        setImplementation(
                Broadcast.class,
                new Broadcast(
                        "",
                        "",
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            double[] t2_val = inputs[2].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] % t2_val[inputs[2].i_of_idx(t2Idx)];
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if (d == 0) {
                                        return 1 / t2_val[inputs[2].i_of_idx(t2Idx)];
                                    } else {
                                        return
                                                -(t1_val[inputs[1].i_of_idx(t1Idx)]
                                                        /
                                                        Math.pow(t2_val[inputs[2].i_of_idx(t2Idx)], 2));
                                    }
                                };
                            }
                        })
        );
        new OperationType(
                "", ((char) 171) + "%", 3, true, false, false, false, false
        );
        new OperationType(
                "", "%" + ((char) 187), 3, true, false, false, false, false
        );
    }

}
