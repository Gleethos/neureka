package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.subtypes.*;

public class Identity extends OperationType {

    public Identity(){

        super("identity", "idy" , 1, false, false, false, true, true);

        set(Activation.class,
                new Activation("output = input;\n",
                        "output = input;\n",
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                            else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                        })
        );
        set(Scalarization.class,
                new Scalarization("output = value;\n",
                        "output = value;\n",
                        null
                )
        );

    }

}
