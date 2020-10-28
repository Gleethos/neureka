package neureka.calculus.backend.implementations.functional;

import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.backend.implementations.AbstractFunctionalOperationTypeImplementation;
import neureka.calculus.backend.operations.OperationType;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.NDIterator;
import org.jetbrains.annotations.Contract;

public class Convolution extends AbstractFunctionalOperationTypeImplementation< Convolution >
{

    public Convolution() {
        super("convolution");
        setSuitabilityChecker( call->1.0f );
    }


    public String getKernelSource() {
        return Neureka.instance().utility().readResource("kernels/convolution_template.cl");
    }

    @Contract(pure = true)
    public static void convolve (
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int d, int i, int end,
            OperationType.TertiaryNDXConsumer operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = t1_src.getNDConf().shape();
        int[] t2Shp = t2_src.getNDConf().shape();
        int rank = t0Shp.length;
        NDIterator t0Idx = NDIterator.of( t0_drn );//t0_drn.idx_of_i( i );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.idx_of_i( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );

        double[] t0_value = t0_drn.value64();

        if (d < 0) {
            while (i < end)//drnSze)
            {//increment on drain accordingly:
                int ri=0;
                while (ri < rank) {
                    if (t1Shp[ri] == t2Shp[ri]) {
                        t1Idx.set( ri, t0Idx.get( ri ) );
                        t2Idx.set( ri, t0Idx.get( ri ) );
                    } else if (t1Shp[ri] > t2Shp[ri]) {
                        t1Idx.set( ri, t0Idx.get( ri ) );
                        t2Idx.set( ri, 0 );
                    } else if (t1Shp[ri] < t2Shp[ri]) {
                        t1Idx.set( ri, 0 );
                        t2Idx.set( ri, t0Idx.get( ri ) );
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
                        if (t1Idx.get( ri ) < t1Shp[ri] && t2Idx.get( ri ) < t2Shp[ri]) {
                            t1Idx.set( ri, t1Idx.get( ri ) + 1 );
                            t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                            if (t1Idx.get( ri ) == t1Shp[ri] || t2Idx.get( ri ) == t2Shp[ri]) {
                                running = (ri != rank - 1);
                                if (t1Shp[ri] == t2Shp[ri]) {
                                    t1Idx.set( ri, t0Idx.get( ri ) );
                                    t2Idx.set( ri, t0Idx.get( ri ) );
                                } else if (t1Shp[ri] > t2Shp[ri]) {
                                    t1Idx.set( ri, t0Idx.get( ri ) );
                                    t2Idx.set( ri, 0 );
                                } else if (t1Shp[ri] < t2Shp[ri]) {
                                    t1Idx.set( ri, 0 );
                                    t2Idx.set( ri, t0Idx.get( ri ) );
                                }
                                ri++;
                            } else incrementing = false;
                        } else ri++;
                    }
                }//setInto _value in drn:
                t0_value[t0Idx.i()] = value;
                //increment on drain:
                t0Idx.increment();
                //NDConfiguration.Utility.increment(t0Idx, t0Shp);

                i++;
            }
        }
        else//---
        {
            // Incrementing if 'i>0' so that all indexes match:
            for(int ii=0; ii<i; ii++) {
                int ri = 0;
                while (ri < rank) {
                    if (t2Idx.get( ri ) == t2Shp[ri]) {
                        t1Idx.set( ri, t0Idx.get( ri ) );
                        t2Idx.set( ri, 0 );
                    } else {
                        t1Idx.set( ri , (t0Shp[ri] > t1Shp[ri])
                                ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                : (t0Idx.get( ri ) + t2Idx.get( ri ))
                        );
                    }
                    ri++;
                }
            }

            // Looping through given range :
            while (i < end) {//increment on drain accordingly:
                int ri=0;
                while (ri < rank) {
                    if (t2Idx.get( ri ) == t2Shp[ri]) {//setting 0
                        t1Idx.set( ri, t0Idx.get( ri ) );
                        t2Idx.set( ri, 0 );
                    } else {
                        t1Idx.set( ri, (t0Shp[ri] > t1Shp[ri])
                                ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                : (t0Idx.get( ri ) + t2Idx.get( ri ))
                        );
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
                            isMatch = (t1Idx.get( rii ) < t1Shp[rii] && t1Idx.get( rii ) >= 0) && isMatch;
                        }
                        value += (isMatch) ? operation.execute(t0Idx, t1Idx, t2Idx) : 0;
                        incrementing = true;
                        ri = 0;
                    } else {//incrementing:
                        if (t2Idx.get( ri ) < t2Shp[ri]) {
                            t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                            if (t2Idx.get( ri ) == t2Shp[ri]) {
                                running = (ri != rank - 1);
                                t1Idx.set( ri, t0Idx.get( ri ) );
                                t2Idx.set( ri, 0 );
                                ri++;
                            } else {
                                t1Idx.set( ri, (t0Shp[ri] > t1Shp[ri])
                                        ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                        : (t0Idx.get( ri ) + t2Idx.get( ri ))
                                );
                                incrementing = false;
                            }
                        } else ri++;
                    }
                }
                //set value in drn:
                t0_value[t0Idx.i()] = value;
                //increment on drain:
                t0Idx.increment();
                //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                i++;
            }
        }
    }



}
