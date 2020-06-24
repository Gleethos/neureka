package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Sinus extends OperationType {

    public Sinus()
    {
        super("sinus", "sin" , 1, false, false, false, true, true);
        setImplementation(Activation.class,
                new Activation(
                        "output = sin(input);\n",
                        "output = cos(input);\n",
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.sin(t1_val[inputs[1].i_of_idx(t1Idx)]);
                            else return (t0Idx, t1Idx, t2Idx) -> Math.cos(t1_val[inputs[1].i_of_idx(t1Idx)]);
                        }
                )
        );

    }


}
