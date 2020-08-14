package neureka.calculus.environment.operations.other;

import neureka.Tsr;
import neureka.acceleration.Device;
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
                                call -> {
                    if ( call.getType().supports(Convolution.class) ) return false;
                    if ( call.getType().identifier().equals(",") ) return false; //Reshape
                    Tsr last = null;
                    for ( Tsr t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                },
                call -> null,
                ( call, goDeeperWith ) -> null,
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[1].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual(false);
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
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
