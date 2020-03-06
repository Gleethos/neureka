package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;

public class Identity extends OperationType {

    public Identity(){

        super("identity", "idy" , true, false, false, true, true,
                "output = input;\n",
                "output = input;\n",
                (inputs, d)->{
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                    else return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)];
                },
                "output = value;\n",
                "output = value;\n",
                null,
                "",
                "",
                null,
                "",
                "",
                null
        );

    }

}
