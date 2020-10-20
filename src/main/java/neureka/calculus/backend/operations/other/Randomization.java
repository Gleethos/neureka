package neureka.calculus.backend.operations.other;

import neureka.Tsr;
import neureka.devices.Device;
import neureka.devices.host.execution.HostExecutor;
import neureka.calculus.Function;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.implementations.functional.Convolution;
import neureka.calculus.backend.implementations.functional.Scalarization;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.backend.operations.OperationType;

import java.util.List;
import java.util.Random;

public class Randomization extends AbstractOperationType
{

    public Randomization()
    {
        super(
                "random", "rand", 1,
                true, false, false, true
        );

        ScalarOperatorCreator< PrimaryNDXConsumer > creator =
                ( inputs, value, d ) -> {
                    return t1Idx -> {
                        int sum = 0;
                        for (int idx : t1Idx) sum += idx;
                        Random dice = new Random();
                        dice.setSeed(Double.doubleToLongBits(value+sum));
                        return dice.nextGaussian();
                    };
                };

        Scalarization scalarization = new Scalarization()
        .setBackwardADAnalyzer( call -> true )
        .setForwardADAnalyzer(
                call -> {
                    if ( call.getType().supports(Convolution.class) ) return false;
                    if ( call.getType().getOperator().equals(",") ) return false; //Reshape
                    Tsr last = null;
                    for ( Tsr t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        )
        .setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    defaultImplementation().supplyADAgentFor( f, call, forward )
        )
        .setCallHock( ( caller, call ) -> null )
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationType.instance("idy") );
                }
        );

        setImplementation(
                Scalarization.class,
                scalarization.setExecutor(
                        HostExecutor.class,
                        new HostExecutor (
                                call -> call.getDevice().getExecutor()
                                        .threaded (
                                                call.getTensor( 0 ).size(),
                                                ( start, end ) ->
                                                        Scalarization.scalarize (
                                                                call.getTensor( 0 ),
                                                                start, end,
                                                                creator.create(
                                                                        call.getTensors(),
                                                                        call.getTensor(1).value64( 0 ),
                                                                        call.getDerivativeIndex()
                                                                )
                                                        )
                                        ),
                                3
                        )
                )
        );

    }

    @Override
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
            return src.get( 0 ).call( inputs, j );
    }
}
