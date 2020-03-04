package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;

public class Gaussian extends OperationType {

    public Gaussian(){

        super("gaussian", "gaus", true, false, false, true, true,
                "output =\n" +
                        "    (float)pow(\n" +
                        "        (float)M_E,\n" +
                        "        -(float)pow(\n" +
                        "            (float)input,\n" +
                        "            (float)2\n" +
                        "        )\n" +
                        "    );\n",
                "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                (inputs, d)->{
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) {
                        return (t0Idx, t1Idx, t2Idx) -> Math.pow(Math.E, -Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], 2));
                    } else {
                        return (t0Idx, t1Idx, t2Idx) -> {
                            double input = t1_val[inputs[1].i_of_idx(t1Idx)];
                            return -2 * input * Math.pow(Math.E, -Math.pow(input, 2));
                        };

                    }
                },
                "",
                "",
                null,
                "",
                "",
                null
        );

    }

}
