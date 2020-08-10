package neureka.calculus.environment.implementations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.Type;
import org.jetbrains.annotations.Contract;

public class Convolution extends AbstractOperationTypeImplementation< Convolution >
{
    public Convolution(){  super();  }

    @Override
    public boolean canHandle(ExecutionCall call) {
        return true;
    }


    public String getKernelSource() {
        return Neureka.instance().utility().readResource("kernels/convolve_template.cl");
    }

    @Contract(pure = true)
    public static void convolve (
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int d, int i, int end,
            Type.TertiaryNDXConsumer operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = t1_src.getNDConf().shape();
        int[] t2Shp = t2_src.getNDConf().shape();
        int rank = t0Shp.length;
        int[] t0Idx = t0_drn.idx_of_i(i);
        int[] t1Idx = new int[rank];
        int[] t2Idx = new int[rank];
        double[] t0_value = t0_drn.value64();

        if (d < 0) {
            while (i < end)//drnSze)
            {//increment on drain accordingly:
                int ri=0;
                while (ri < rank) {
                    if (t1Shp[ri] == t2Shp[ri]) {
                        t1Idx[ri] = t0Idx[ri];
                        t2Idx[ri] = t0Idx[ri];
                    } else if (t1Shp[ri] > t2Shp[ri]) {
                        t1Idx[ri] = t0Idx[ri];
                        t2Idx[ri] = 0;
                    } else if (t1Shp[ri] < t2Shp[ri]) {
                        t1Idx[ri] = 0;
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
                        if (t1Idx[ri] < t1Shp[ri] && t2Idx[ri] < t2Shp[ri]) {
                            t1Idx[ri]++;
                            t2Idx[ri]++;
                            if (t1Idx[ri] == t1Shp[ri] || t2Idx[ri] == t2Shp[ri]) {
                                running = (ri != rank - 1);
                                if (t1Shp[ri] == t2Shp[ri]) {
                                    t1Idx[ri] = t0Idx[ri];
                                    t2Idx[ri] = t0Idx[ri];
                                } else if (t1Shp[ri] > t2Shp[ri]) {
                                    t1Idx[ri] = t0Idx[ri];
                                    t2Idx[ri] = 0;
                                } else if (t1Shp[ri] < t2Shp[ri]) {
                                    t1Idx[ri] = 0;
                                    t2Idx[ri] = t0Idx[ri];
                                }
                                ri++;
                            } else incrementing = false;
                        } else ri++;
                    }
                }//setInto _value in drn:
                t0_value[t0_drn.i_of_idx(t0Idx)] = value;
                //increment on drain:
                Tsr.Utility.Indexing.increment(t0Idx, t0Shp);

                i++;
            }
        }
        else//---
        {
            // Incrementing if 'i>0' so that all indexes match:
            for(int ii=0; ii<i; ii++) {
                int ri = 0;
                while (ri < rank) {
                    if (t2Idx[ri] == t2Shp[ri]) {
                        t1Idx[ri] = t0Idx[ri];
                        t2Idx[ri] = 0;
                    } else {
                        t1Idx[ri] = (t0Shp[ri] > t1Shp[ri])
                                ? (t0Idx[ri] - t2Idx[ri])
                                : (t0Idx[ri] + t2Idx[ri]);
                    }
                    ri++;
                }
            }

            // Looping through given range :
            while (i < end) {//increment on drain accordingly:
                int ri=0;
                while (ri < rank) {
                    if (t2Idx[ri] == t2Shp[ri]) {//setting 0
                        t1Idx[ri] = t0Idx[ri];
                        t2Idx[ri] = 0;
                    } else {
                        t1Idx[ri] = (t0Shp[ri] > t1Shp[ri])
                                ? (t0Idx[ri] - t2Idx[ri])
                                : (t0Idx[ri] + t2Idx[ri]);
                    }
                    ri++;
                }
                //----------
                double value = 0;
                boolean running = true;
                boolean incrementing = false;
                while (running) {
                    ri = (ri == rank) ? 0 : ri;
                    if (!incrementing) {// := testing for match and applying operation:
                        boolean isMatch = true;
                        for (int rii = 0; rii < rank; rii++) {
                            isMatch = (t1Idx[rii] < t1Shp[rii] && t1Idx[rii] >= 0) && isMatch;
                        }
                        value += (isMatch) ? operation.execute(t0Idx, t1Idx, t2Idx) : 0;
                        incrementing = true;
                        ri = 0;
                    } else {//incrementing:
                        if (t2Idx[ri] < t2Shp[ri]) {
                            t2Idx[ri]++;
                            if (t2Idx[ri] == t2Shp[ri]) {
                                running = (ri != rank - 1);
                                t1Idx[ri] = t0Idx[ri];
                                t2Idx[ri] = 0;
                                ri++;
                            } else {
                                t1Idx[ri] = (t0Shp[ri] > t1Shp[ri])
                                        ? (t0Idx[ri] - t2Idx[ri])
                                        : (t0Idx[ri] + t2Idx[ri]);
                                incrementing = false;
                            }
                        } else ri++;
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
