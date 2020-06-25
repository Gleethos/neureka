package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Power extends OperationType
{

    private final static OperatorCreator _creator = (inputs, d)->{
        double[] t1_val = inputs[1].value64();
        double[] t2_val = inputs[2].value64();
        if (d < 0) {
            return (t0Idx, t1Idx, t2Idx) -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], t2_val[inputs[2].i_of_idx(t2Idx)]);
        } else {
            return (t0Idx, t1Idx, t2Idx) -> {
                if (d == 0) {
                    return t2_val[inputs[2].i_of_idx(t2Idx)]
                            * Math.pow(
                            t1_val[inputs[1].i_of_idx(t1Idx)],
                            t2_val[inputs[2].i_of_idx(t2Idx)] - 1
                    );
                } else {
                    return Math.pow(
                            t1_val[inputs[1].i_of_idx(t1Idx)],
                            t2_val[inputs[2].i_of_idx(t2Idx)]
                    ) * Math.log(t1_val[inputs[1].i_of_idx(t1Idx)]);
                }
            };
        }
    };

    public Power()
    {
        super("power", "^", -1, true, false, false, false, false);

        setImplementation(Scalarization.class,
                new Scalarization(
                        "output = pow(input1, value);",
                        "if ( d==0 ) {                                     \n" +
                                "    output = value * pow(input1, value-(float)1 );   \n" +
                                "} else {                                             \n" +
                                "    output = pow(input1, value) * log(value);        \n" +
                                "}",
                        ( inputs, value, d )->{
                            double[] t1_val = inputs[1].value64();
                            if (d < 0) {
                                return ( t1Idx ) -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value);
                            } else {
                                if(d==0){
                                    return ( t1Idx ) -> value*Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value-1);
                                } else {
                                    return ( t1Idx ) -> Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value)*Math.log(value);
                                }
                            }
                        })
        );
        setImplementation(Broadcast.class,
                new Broadcast(
                        "value += pow(src1, src2);",
                        "if(d==0){\n" +
                                "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                                "} else {\n" +
                                "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                                "}",
                        _creator
                )
        );
        setImplementation(Operation.class,
                new Operation(
                        "output = pow(input1, input2);",
                        "if(d==0){\n" +
                                "    output = input2 * pow(input1, input2-1.0f);\n" +
                                "} else {\n" +
                                "    output = pow(input1, input2) * log(input1);\n" +
                                "}",
                        _creator
                )
        );


        new OperationType("inv_power_left", ((char)171)+"^", 3, true, false, false, false, false);
        new OperationType("inv_power_right", "^" + ((char) 187), 3, true, false, false, false, false);

        // Convolution:

        new OperationType(
                "power", "p", 2, true, false, true, false, false
        ).setImplementation(Convolution.class,
                new Convolution(
                        "value += pow(src1, src2);",
                        "if(d==0){\n" +
                                "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                                "} else {\n" +
                                "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                                "}",
                        null
                )
        );
        new OperationType("", ((char) 171) + "p", 3, true, false, true, false, false);
        new OperationType("", "p" + ((char) 187), 3, true, false, true, false, false);




    }

}
