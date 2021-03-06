package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.dtype.NumericType;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

public class Convolution extends AbstractFunctionalAlgorithm< Convolution >
{

    public Convolution() {
        super("convolution");
        setIsSuitableFor( call ->
                call.validate()
                .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                .estimation()
        );
    }


    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/convolution_template.cl");
    }

    @Contract(pure = true)
    public static void convolve (
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int d, int i, int end,
            Operation.TertiaryNDIConsumer operation
    ) {
        if ( d < 0 ) _convolve(t0_drn, t1_src, t2_src, i, end, operation);
        else _deConvolve(t0_drn, t1_src, t2_src, i, end, operation);
    }

    @Contract(pure = true)
    public static void _convolve (
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int i, int end,
            Operation.TertiaryNDIConsumer operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        double[] t0_value = t0_drn.value64();

        while (i < end)//drnSze)
        {//increment on drain accordingly:
            int ri=0;
            while (ri < rank) {
                if (t1Idx.shape( ri ) == t2Idx.shape( ri )) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, t0Idx.get( ri ) );
                } else if (t1Idx.shape( ri ) > t2Idx.shape( ri )) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, 0 );
                } else if (t1Idx.shape( ri ) < t2Idx.shape( ri )) {
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
            while ( running ) {
                ri = (ri == rank) ? 0 : ri;
                if (!incrementing) {
                    value += operation.execute( t0Idx, t1Idx, t2Idx );
                    incrementing = true;
                    ri = 0;
                } else { // incrementing:
                    if (t1Idx.get( ri ) < t1Idx.shape( ri ) && t2Idx.get( ri ) < t2Idx.shape( ri )) {
                        t1Idx.set( ri, t1Idx.get( ri ) + 1 );
                        t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                        if (t1Idx.get( ri ) == t1Idx.shape( ri ) || t2Idx.get( ri ) == t2Idx.shape( ri )) {
                            running = (ri != rank - 1);
                            if (t1Idx.shape( ri ) == t2Idx.shape( ri )) {
                                t1Idx.set( ri, t0Idx.get( ri ) );
                                t2Idx.set( ri, t0Idx.get( ri ) );
                            } else if (t1Idx.shape( ri ) > t2Idx.shape( ri )) {
                                t1Idx.set( ri, t0Idx.get( ri ) );
                                t2Idx.set( ri, 0 );
                            } else if (t1Idx.shape( ri ) < t2Idx.shape( ri )) {
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

    private static void _deConvolve(
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int i, int end,
            Operation.TertiaryNDIConsumer operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        double[] t0_value = t0_drn.value64();

        // Incrementing if 'i>0' so that all indexes match:
        for(int ii=0; ii<i; ii++ ) {
            int ri = 0;
            while (ri < rank) {
                if (t2Idx.get( ri ) == t2Idx.shape( ri )) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, 0 );
                } else {
                    t1Idx.set( ri , (t0Idx.shape( ri ) > t1Idx.shape( ri ))
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
                if (t2Idx.get( ri ) == t2Idx.shape( ri )) {//setting 0
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, 0 );
                } else {
                    t1Idx.set( ri, (t0Idx.shape( ri ) > t1Idx.shape( ri ))
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
                    for ( int rii = 0; rii < rank; rii++ ) {
                        isMatch = (t1Idx.get( rii ) < t1Idx.shape( rii ) && t1Idx.get( rii ) >= 0) && isMatch;
                    }
                    value += (isMatch) ? operation.execute( t0Idx, t1Idx, t2Idx ) : 0;
                    incrementing = true;
                    ri = 0;
                } else { // incrementing:
                    if (t2Idx.get( ri ) < t2Idx.shape( ri )) {
                        t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                        if (t2Idx.get( ri ) == t2Idx.shape( ri )) {
                            running = (ri != rank - 1);
                            t1Idx.set( ri, t0Idx.get( ri ) );
                            t2Idx.set( ri, 0 );
                            ri++;
                        } else {
                            t1Idx.set( ri, (t0Idx.shape( ri ) > t1Idx.shape( ri ))
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

    // ARRAY BASED CONVOLUTION


    @Contract(pure = true)
    public static void convolve (
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int d, int i, int end,
            Operation.TertiaryNDAConsumer operation
    ) {
        if ( d < 0 ) _convolve(t0_drn, t1_src, t2_src, i, end, operation);
        else _deConvolve(t0_drn, t1_src, t2_src, i, end, operation);
    }


    @Contract(pure = true)
    private static void _convolve (
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int i, int end,
            Operation.TertiaryNDAConsumer operation
    ) {
        NDConfiguration ndc0 = t0_drn.getNDConf();
        NDConfiguration ndc1 = t1_src.getNDConf();
        NDConfiguration ndc2 = t2_src.getNDConf();
        int[] t0Shp = ndc0.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = ndc1.shape();
        int[] t2Shp = ndc2.shape();
        int rank = t0Shp.length;
        int[] t0Idx = ndc0.indicesOfIndex( i );
        int[] t1Idx = new int[ rank ];
        int[] t2Idx = new int[ rank ];
        double[] t0_value = (double[]) t0_drn.getData();

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
                    value += operation.execute( t0Idx, t1Idx, t2Idx );
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
            t0_value[ndc0.indexOfIndices(t0Idx)] = value;
            //increment on drain:
            NDConfiguration.Utility.increment(t0Idx, t0Shp);

            i++;
        }
    }

    @Contract(pure = true)
    private static void _deConvolve (
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int i, int end,
            Operation.TertiaryNDAConsumer operation
    ) {
        NDConfiguration ndc0 = t0_drn.getNDConf();
        NDConfiguration ndc1 = t1_src.getNDConf();
        NDConfiguration ndc2 = t2_src.getNDConf();
        int[] t0Shp = ndc0.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = ndc1.shape();
        int[] t2Shp = ndc2.shape();
        int rank = t0Shp.length;
        int[] t0Idx = ndc0.indicesOfIndex( i );
        int[] t1Idx = new int[ rank ];
        int[] t2Idx = new int[ rank ];
        double[] t0_value = (double[]) t0_drn.getData();

        // Incrementing if 'i>0' so that all indexes match:
        for(int ii=0; ii<i; ii++ ) {
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
                    for ( int rii = 0; rii < rank; rii++ ) {
                        isMatch = (t1Idx[rii] < t1Shp[rii] && t1Idx[rii] >= 0) && isMatch;
                    }
                    value += (isMatch) ? operation.execute( t0Idx, t1Idx, t2Idx ) : 0;
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
            t0_value[ndc0.indexOfIndices(t0Idx)] = value;
            //increment on drain:
            NDConfiguration.Utility.increment(t0Idx, t0Shp);
            i++;
        }

    }

}
