package neureka.calculus.backend.implementations.functional;

import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.backend.implementations.AbstractFunctionalOperationTypeImplementation;
import neureka.calculus.backend.operations.OperationType;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.NDIterator;
import org.jetbrains.annotations.Contract;

public class Scalarization extends AbstractFunctionalOperationTypeImplementation< Scalarization >
{

    public Scalarization(){
        super("scalarization");
        setSuitabilityChecker(call->1.0f);
    }


    public String getKernelSource(){
        return Neureka.instance().utility().readResource("kernels/scalarization_template.cl");
    }


    @Contract(pure = true)
    public static void scalarize (
            Tsr t0_drn,
            int i, int end,
            OperationType.PrimaryNDIConsumer operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();
        NDIterator t0Idx = NDIterator.of( t0_drn ); //t0_drn.idx_of_i( i );
        t0Idx.set( t0_drn.idx_of_i( i ) );
        double[] t0_value = t0_drn.value64();
        while (i < end) // increment on drain accordingly:
        {
            // setInto _value in drn:
            t0_value[t0Idx.i()] = operation.execute( t0Idx );
            // increment on drain:
            t0Idx.increment();
            //NDConfiguration.Utility.increment(t0Idx, t0Shp);
            i++;
        }
    }

    @Contract(pure = true)
    public static void scalarize (
            Tsr t0_drn,
            int i, int end,
            OperationType.PrimaryNDXConsumer operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();
        int[] t0Idx = t0_drn.idx_of_i( i );
        double[] t0_value = t0_drn.value64();
        while (i < end) // increment on drain accordingly:
        {
            // setInto _value in drn:
            t0_value[t0_drn.i_of_idx(t0Idx)] = operation.execute( t0Idx );
            // increment on drain:
            NDConfiguration.Utility.increment(t0Idx, t0Shp);
            i++;
        }
    }


}