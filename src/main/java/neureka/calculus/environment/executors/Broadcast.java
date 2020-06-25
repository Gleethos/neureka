package neureka.calculus.environment.executors;

import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.environment.Type;
import org.jetbrains.annotations.Contract;

public class Broadcast extends AbstractOperationTypeImplementation<Broadcast, Type.DefaultOperatorCreator<Type.TertiaryNDXConsumer>>
{
    public Broadcast(String strBroadcast, String strDeriviation, Type.DefaultOperatorCreator creator){
        super(strBroadcast, strDeriviation, creator);
    }

    @Override
    public boolean canHandle(ExecutionCall call)
    {
        int maxRank = 0;
        for ( Tsr t : call.getTensors() ) if( t!=null && t.rank()>maxRank) maxRank = t.rank();

        for ( int i = 0; i < maxRank; i++ )
        {
            int currentDim = -1;
            for( Tsr t : call.getTensors() )
            {
                if( t!=null && i<t.rank() ) {
                    if ( currentDim == -1 ) currentDim = t.shape(i);
                    else if ( currentDim!=t.shape(i) && currentDim!=1 && t.shape(i)!=1 ) return false;
                }
            }
        }
        return true;
    }

    public String getKernelSource(){
        return Neureka.instance().utility().readResource("kernels/broadcast_template.cl");
    }

    @Contract(pure = true)
    public static void broadcast(
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int d, int i, int end,
            Type.TertiaryNDXConsumer operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = t1_src.getNDConf().shape();
        int[] t2Shp = (t2_src != null) ? t2_src.getNDConf().shape() : t1Shp;
        int rank = t0Shp.length;
        int[] t0Idx = t0_drn.idx_of_i(i);
        int[] t1Idx = new int[rank];
        int[] t2Idx = new int[rank];
        double[] t0_value = t0_drn.value64();
        if (d < 0) {
            while (i < end) {//increment on drain accordingly:
                int ri = 0;
                while (ri < rank) {
                    if (t1Shp[ri] == t2Shp[ri]) {//Equal shapes -> out index is t1 & t2 index!for this ri
                        t1Idx[ri] = t0Idx[ri];
                        t2Idx[ri] = t0Idx[ri];
                    } else if (t1Shp[ri] > t2Shp[ri]) {//Current shape axis of t2 must be 1 !
                        t1Idx[ri] = t0Idx[ri];
                        t2Idx[ri] = 0;//...therefore it can be set to 0!
                    } else if (t1Shp[ri] < t2Shp[ri]) {//same principle:
                        t1Idx[ri] = 0;
                        t2Idx[ri] = t0Idx[ri];
                    }
                    ri++;
                }
                //----------
                //setInto _value in drn:
                t0_value[t0_drn.i_of_idx(t0Idx)] = operation.execute(t0Idx, t1Idx, t2Idx);
                //increment on drain:
                Tsr.Utility.Indexing.increment(t0Idx, t0Shp);
                i++;
            }
        }
        else//---//Note: src2 is now former drain!
        {
            while (i < end) {//increment on drain accordingly:
                int ri = 0;
                while (ri < rank) {
                    if (t0Shp[ri] == t1Shp[ri]) {
                        t1Idx[ri] = t0Idx[ri];//all shapes are equal -> shape index can be inherited from origin!
                        t2Idx[ri] = t0Idx[ri];
                    } else if (t0Shp[ri] > t1Shp[ri]) {
                        t1Idx[ri] = 0;//Current origin index is larger: index can be inherited!
                        t2Idx[ri] = t0Idx[ri];
                    }
                    ri++;
                }
                //----------
                // multiplication:
                double value = 0;
                boolean running = true;
                boolean incrementing = false;
                while (running) {
                    ri = (ri == rank) ? 0 : ri;
                    if (!incrementing) {
                        value += operation.execute(t0Idx, t1Idx, t2Idx);
                        incrementing = true;
                        ri = 0;
                    } else {//incrementing:
                        if (t0Shp[ri] < t1Shp[ri]) {//Only if origin shape is smaller than handle and drain!
                            t1Idx[ri]++;
                            t2Idx[ri]++;
                            if (t1Idx[ri] == t1Shp[ri]) {
                                t1Idx[ri] = 0;
                                t2Idx[ri] = 0;
                                running = (ri != rank - 1);
                                ri++;
                            } else {
                                incrementing = false;//return to calculation!
                            }
                        } else {
                            running = (ri != rank - 1);
                            ri++;
                        }
                    }
                }
                //set value in drn:
                t0_value[t0_drn.i_of_idx(t0Idx)] = value;
                //increment on drain:
                Tsr.Utility.Indexing.increment(t0Idx, t0Shp);
                i++;
            }
        }
    }


}
