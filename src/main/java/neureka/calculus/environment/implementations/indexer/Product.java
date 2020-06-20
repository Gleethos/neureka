package neureka.calculus.environment.implementations.indexer;

import neureka.calculus.environment.OperationType;

public class Product extends OperationType {

    public Product(){
        super("product", "prod", false,  true, false, true, true);
        set(Activation.class,
                new Activation("output = input;",
                        "output = 1;",
                        null
                )
        );
        set(Broadcast.class,
                new Broadcast(
                        "",
                        "",
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            double[] t2_val = inputs[2].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    if (d == 0) return t2_val[inputs[2].i_of_idx(t2Idx)];
                                    else return t1_val[inputs[1].i_of_idx(t1Idx)];
                                };
                            }
                        })
        );


    }




}
