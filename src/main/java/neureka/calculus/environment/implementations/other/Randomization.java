package neureka.calculus.environment.implementations.other;

import neureka.calculus.environment.OperationType;

import java.util.Random;

public class Randomization extends OperationType{

    public Randomization(){

        // Operations (auto broadcast):
        super(
                "random", "rand", false, false, false, false, false,
                null,
                new Scalarization("output = pow(input1, value);",
                        "if(d==0){\n" +
                                "    output = value * pow(input1, value-(float)1 );\n" +
                                "} else {\n" +
                                "    output = pow(input1, value) * log(value);\n" +
                                "}",
                        (inputs, value, d)->{
                            Random dice = new Random();
                            dice.setSeed(Double.doubleToLongBits(value));
                            double[] t1_val = inputs[1].value64();
                            return (t0Idx, t1Idx, t2Idx) -> dice.nextGaussian();
                            //Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value);
                        }),
                null,
                null,
                null
        );

    }

}
