package neureka.calculus.environment.implementations.function;

import neureka.calculus.environment.OperationType;


public class Ligmoid extends OperationType {

    public Ligmoid(){
        super("ligmoid", "lig" , true, false, false, true, true,
                "output = \n" +
                        "(\n" +
                        "        (float) log(\n" +
                        "            1+pow(\n" +
                        "                (float)\n" +
                        "                M_E,\n" +
                        "                (float)\n" +
                        "                input\n" +
                        "            )\n" +
                        "        )\n" +
                        "    );",
                "output =\n" +
                        "    1 /\n" +
                        "        (1 + (float) pow(\n" +
                        "                (float)M_E,\n" +
                        "                (float)input\n" +
                        "            )\n" +
                        "        );\n",
                (inputs, d)->{
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return (t0Idx, t1Idx, t2Idx) -> Math.log(1 + Math.pow(Math.E, t1_val[inputs[1].i_of_idx(t1Idx)]));
                    else return (t0Idx, t1Idx, t2Idx) -> 1 / (1 + Math.pow(Math.E, -t1_val[inputs[1].i_of_idx(t1Idx)]));
                },
                "",
                "",
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
