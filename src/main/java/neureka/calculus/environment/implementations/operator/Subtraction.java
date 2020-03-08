package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;

public class Subtraction extends OperationType {

    public Subtraction(){

        super(
                "subtract", "-", false, false, false, false, false,

                null,
                new Scalarization("output = input1 - value;\n",
                        "if(d==0){\n" +//drn and src2 switch:
                                "    output = 1;\n" +
                                "} else {\n" +
                                "    output = -1;" +
                                "}",
                        (inputs, value, d) -> {
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] - value;
                            } else {
                                if (d == 0) return (t0Idx, t1Idx, t2Idx) -> 1;
                                else return (t0Idx, t1Idx, t2Idx) -> -1;
                            }
                        })
                ,
                null,
                new Broadcast("",
                        "",
                        (inputs, d) -> {
                            double[] t1_val = inputs[1].value64();
                            double[] t2_val = inputs[2].value64();
                            if (d < 0) {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    return t1_val[inputs[1].i_of_idx(t1Idx)] - t2_val[inputs[2].i_of_idx(t2Idx)];
                                };
                            } else {
                                return (t0Idx, t1Idx, t2Idx) -> {
                                    return (d == 0) ? 1.0 : -1.0;
                                };
                            }
                        }),
                null

        );
        new OperationType(
                "", ((char) 171) + "-", false, false, false, false, false,
                null, null, null, null, null
        );
        new OperationType(
                "", "-" + ((char) 187), false, false, false, false, false,
                null, null, null, null, null
        );

        // Convolution:


        new OperationType(
                "", "s", false, false, true, false, false,
                null, null, null, null, null
        );
        new OperationType(
                "", ((char) 171) + "s", false, false, true, false, false,
                null, null, null, null, null
        );
        new OperationType(
                "", "s" + ((char) 187), false, false, true, false, false,
                null, null, null, null, null
        );


    }


}
