package neureka.calculus.environment.implementations.indexer;

import neureka.calculus.environment.OperationType;

public class Summation extends OperationType {

    public Summation(){
        super("summation", "sum" , false, false, true, false, true, true,
                new Activation("output = input;",
                        "output = 1;",
                        null)
                ,
                null,
                null,
                new Broadcast(
                        "",
                "",
                (inputs, d)->{
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if (d < 0) {
                        return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] + t2_val[inputs[2].i_of_idx(t2Idx)];
                    } else {
                        return (t0Idx, t1Idx, t2Idx) -> 1.0;
                    }
                }),
                null
        );
    }

}
