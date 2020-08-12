package neureka.calculus.environment.operations.other;

import neureka.acceleration.host.execution.HostExecutor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.*;

import java.util.Random;

public class Randomization extends OperationType
{

    public Randomization()
    {
        super(
                "random", "rand", 1,
                true, false, false, false, false
        );

        ScalarOperatorCreator< PrimaryNDXConsumer > creator =
                ( inputs, value, d ) -> {
                    //double[] t1_val = inputs[1].value64();
                    return t1Idx -> {
                        int sum = 0;
                        for (int idx : t1Idx) sum += idx;
                        Random dice = new Random();
                        dice.setSeed(Double.doubleToLongBits(value+sum));
                        return dice.nextGaussian();
                    };
                };

        Scalarization scalarization = new Scalarization(
                call -> true,
                ( call, goDeeperWith ) -> null
        );

        setImplementation(
                Scalarization.class,
                scalarization.setExecutor(
                        HostExecutor.class,
                        new HostExecutor (
                                call -> call.getDevice().getExecutor()
                                        .threaded (
                                                call.getTensor(0).size(),
                                                ( start, end ) ->
                                                        Scalarization.scalarize (
                                                                call.getTensor(0),
                                                                start, end,
                                                                creator.create(
                                                                        call.getTensors(),
                                                                        call.getTensor(1).value64(0),
                                                                        call.getDerivativeIndex()
                                                                )
                                                        )
                                        ),
                                3
                        )
                )
        );

    }

}
