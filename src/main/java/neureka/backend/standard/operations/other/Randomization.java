package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;

import java.util.List;
import java.util.Random;

public class Randomization extends AbstractOperation
{

    public Randomization()
    {
        super(
                "random", "rand", 1,
                true, false, false, true
        );

        ScalarOperatorCreator< PrimaryNDIConsumer > creator =
                ( inputs, value, d ) -> {
                    return t1Idx -> {
                        int sum = 0;
                        int[] idx = t1Idx.get();
                        for ( int i : idx) sum += i;
                        Random dice = new Random();
                        dice.setSeed(Double.doubleToLongBits(value+sum));
                        return dice.nextGaussian();
                    };
                };

        ScalarOperatorCreator< PrimaryNDXConsumer > creatorX =
                ( inputs, value, d ) -> {
                    return t1Idx -> {
                        int sum = 0;
                        for ( int idx : t1Idx) sum += idx;
                        Random dice = new Random();
                        dice.setSeed(Double.doubleToLongBits(value+sum));
                        return dice.nextGaussian();
                    };
                };

        Scalarization scalarization = new Scalarization()
        .setBackwardADAnalyzer( call -> true )
        .setForwardADAnalyzer(
                call -> {
                    if ( call.getOperation().supports(Convolution.class) ) return false;
                    if ( call.getOperation().getOperator().equals(",") ) return false; //Reshape
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        )
        .setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
        )
        .setCallHook( (caller, call ) -> null )
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return new ExecutionCall( call.getDevice(), new Tsr[]{tsrs[offset], tsrs[1+offset]}, -1, OperationContext.get().instance("idy") );
                }
        )
        .build();

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call -> call.getDevice().getExecutor()
                                        .threaded (
                                                call.getTensor( 0 ).size(),
                                                (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                ? ( start, end ) ->
                                                        Scalarization.scalarize (
                                                                call.getTensor( 0 ),
                                                                start, end,
                                                                creatorX.create(
                                                                        call.getTensors(),
                                                                        call.getTensor( 1 ).value64( 0 ),
                                                                        call.getDerivativeIndex()
                                                                )
                                                        )
                                                : ( start, end ) ->
                                                        Scalarization.scalarize (
                                                            call.getTensor( 0 ),
                                                            start, end,
                                                            creator.create(
                                                                    call.getTensors(),
                                                                    call.getTensor( 1 ).value64( 0 ),
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
