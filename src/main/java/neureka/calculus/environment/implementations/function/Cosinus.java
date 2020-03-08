package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;

public class Cosinus extends OperationType {

    public Cosinus(){

        super(
                "cosinus", "cos" , true, false, false, true, true,
                new Activation("output = cos(input);\n",
                        "output = -sin(input);\n",
                        (inputs, d)->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.cos(t1_val[inputs[1].i_of_idx(t1Idx)]);
                            else return (t0Idx, t1Idx, t2Idx) -> -Math.sin(t1_val[inputs[1].i_of_idx(t1Idx)]);
                        }),
                null,
                null,
                null,
                null
        );


    }



}
