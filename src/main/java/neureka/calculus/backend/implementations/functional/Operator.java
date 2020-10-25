package neureka.calculus.backend.implementations.functional;

import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.backend.implementations.AbstractFunctionalOperationTypeImplementation;
import neureka.calculus.backend.operations.OperationType;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

public class Operator extends AbstractFunctionalOperationTypeImplementation<Operator>
{
    public Operator() {
        super("operator");
        setSuitabilityChecker(
                call -> {
                    int size = ( call.getTensors()[ 0 ] == null ) ? call.getTensors()[ 1 ].size() : call.getTensors()[ 0 ].size();
                    for ( Tsr t : call.getTensors() ) if ( t!=null && t.size() != size ) return 0.0f;
                    return 1.0f;
                }
        );
    }

    public String getKernelSource(){
        return Neureka.instance().utility().readResource("kernels/operator_template.cl");
    }


    @Contract(pure = true)
    public static void operate(
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int d, int i, int end,
            OperationType.PrimaryNDXConsumer operation
    ) {
        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() ) {
            ((double[])t0_drn.getValue())[ 0 ] = operation.execute( new int[t0_drn.rank()] );
        } else {
            int[] t0Shp = t0_drn.getNDConf().shape(); // Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
            int[] t0Idx = t0_drn.idx_of_i( i );
            double[] t0_value = t0_drn.value64();
            while (i < end) {//increment on drain accordingly:
                //setInto _value in drn:
                t0_value[t0_drn.i_of_idx(t0Idx)] = operation.execute( t0Idx );
                //increment on drain:
                NDConfiguration.Utility.increment(t0Idx, t0Shp);
                i++;
            }
        }
    }


}