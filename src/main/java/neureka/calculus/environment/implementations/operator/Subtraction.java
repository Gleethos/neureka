package neureka.calculus.environment.implementations.operator;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

public class Subtraction extends OperationType {

    private static final OperatorCreator _creator =
            (inputs, d) -> {
                double[] t1_val = inputs[1].value64();
                double[] t2_val = inputs[2].value64();
                if (d < 0) {
                    return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] - t2_val[inputs[2].i_of_idx(t2Idx)];
                } else return (t0Idx, t1Idx, t2Idx) -> (d == 0) ? 1.0 : -1.0;
            };

    public Subtraction(){

        super(
                "subtract", "-", -1, true, false, false, false, false
        );

        //_____________________
        // DEFAULT OPERATION :

        setImplementation(Operation.class,
                new Operation(
                        "output = input1 - input2;  \n",
                        "if(d==0){                 \n" +//drn and src2 switch:
                                "    output = 1;              \n" +
                                "} else {                     \n" +
                                "    output = -1;               " +
                                "}",
                        _creator
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        setImplementation(Scalarization.class,
        new Scalarization(
                "output = input1 - value;\n",
                "if(d==0){     \n" +//drn and src2 switch:
                        "    output = 1;  \n" +
                        "} else {         \n" +
                        "    output = -1;   " +
                        "}",
                (inputs, value, d) -> {
                    double[] t1_val = inputs[1].value64();
                    if (d < 0) return t1Idx -> t1_val[inputs[1].i_of_idx(t1Idx)] - value;
                    else {
                        if (d == 0) return t1Idx -> 1; else return t1Idx -> -1;
                    }
        }));

        //________________
        // BROADCASTING :

        setImplementation(Broadcast.class,
                new Broadcast(
                        "value = src1 - src2;   \n",
                        "if(d==0){              \n" +//drn and src2 switch:
                                "    value += 1 * drain;   \n" +
                                "} else {                  \n" +
                                "    value += -1 * drain;    " +
                                "}",
                        _creator
                )
        );

        //______________________
        // RELATED OPERATIONS :

        new OperationType(
                "", ((char) 171) + "-", 3, true, false, false, false, false
        );
        new OperationType(
                "", "-" + ((char) 187), 3, true, false, false, false, false
        );

        // Convolution:


        new OperationType(
                "", "s", 2, true, false, true, false, false
        );
        new OperationType(
                "", ((char) 171) + "s", 3, true, false, true, false, false
        );
        new OperationType(
                "", "s" + ((char) 187), 3, true, false, true, false, false
        );


    }


}
