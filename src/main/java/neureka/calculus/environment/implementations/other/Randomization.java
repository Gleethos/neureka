package neureka.calculus.environment.implementations.other;

import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.*;

import java.util.Random;

public class Randomization extends OperationType{

    public Randomization(){

        // Operations (auto broadcast):
        super(
                "random", "rand", 1, true, false, false, false, false
        );
        setImplementation(Scalarization.class, new Scalarization(
                (inputs, value, d)->{
                    Random dice = new Random();
                    dice.setSeed(Double.doubleToLongBits(value));
                    double[] t1_val = inputs[1].value64();
                    return ( t1Idx ) -> dice.nextGaussian();
                    //Math.pow(t1_val[inputs[1].i_of_idx(t1Idx)], value);
                }));

    }

}
