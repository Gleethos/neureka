package neureka.calculus.environment.executors;

import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.environment.Type;
import org.jetbrains.annotations.Contract;

public class Scalarization extends AbstractOperationTypeImplementation<Scalarization, Type.ScalarOperatorCreator<Type.PrimaryNDXConsumer>>
{
    public Scalarization(String strScalarized, String strDeriviation, Type.ScalarOperatorCreator<Type.PrimaryNDXConsumer> creator)
    {
        super(strScalarized, strDeriviation, creator);
    }

    @Override
    public boolean canHandle(ExecutionCall call)
    {
        return true;
    }


    public String getKernelSource(){
        return Neureka.instance().utility().readResource("kernels/scalar_template.cl");
    }


    @Contract(pure = true)
    public static void scalarize (
            Tsr t0_drn,
            int i, int end,
            Type.PrimaryNDXConsumer operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();
        int[] t0Idx = t0_drn.idx_of_i(i);
        double[] t0_value = t0_drn.value64();
        while (i < end) // increment on drain accordingly:
        {
            // setInto _value in drn:
            t0_value[t0_drn.i_of_idx(t0Idx)] = operation.execute( t0Idx );
            // increment on drain:
            Tsr.Utility.Indexing.increment(t0Idx, t0Shp);
            i++;
        }
    }



}