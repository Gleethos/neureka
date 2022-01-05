package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

public class Convolution extends AbstractFunctionalAlgorithm< Convolution >
{

    public Convolution() {
        super("convolution");
        setIsSuitableFor( call ->
                call.validate()
                .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                .basicSuitability()
        );
    }


    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/convolution_template.cl");
    }

    @Contract(pure = true)
    public static void convolve (
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int d, int i, int end,
            Operation.TertiaryNDIConsumer operation
    ) {
        if ( d < 0 ) _convolve(t0_drn, t1_src, t2_src, i, end, operation);
        else _deConvolve(t0_drn, t1_src, t2_src, i, end, operation);
    }

    @Contract(pure = true)
    public static void _convolve (
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            Operation.TertiaryNDIConsumer operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        double[] t0_value = t0_drn.getDataAs( double[].class );

        while ( i < end )
        {//increment on drain accordingly:
            int ri = 0;
            while ( ri < rank ) {
                if ( t1Idx.shape( ri ) == t2Idx.shape( ri ) ) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, t0Idx.get( ri ) );
                } else if ( t1Idx.shape( ri ) > t2Idx.shape( ri ) ) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, 0 );
                } else if ( t1Idx.shape( ri ) < t2Idx.shape( ri ) ) {
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
                ri = ( ri == rank ) ? 0 : ri;
                if ( !incrementing ) {
                    value += operation.execute( t0Idx, t1Idx, t2Idx );
                    incrementing = true;
                    ri = 0;
                } else { // incrementing:
                    if ( t1Idx.get( ri ) < t1Idx.shape( ri ) && t2Idx.get( ri ) < t2Idx.shape( ri ) ) {
                        t1Idx.set( ri, t1Idx.get( ri ) + 1 );
                        t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                        if ( t1Idx.get( ri ) == t1Idx.shape( ri ) || t2Idx.get( ri ) == t2Idx.shape( ri )) {
                            running = (ri != rank - 1);
                            if ( t1Idx.shape( ri ) == t2Idx.shape( ri ) ) {
                                t1Idx.set( ri, t0Idx.get( ri ) );
                                t2Idx.set( ri, t0Idx.get( ri ) );
                            } else if ( t1Idx.shape( ri ) > t2Idx.shape( ri ) ) {
                                t1Idx.set( ri, t0Idx.get( ri ) );
                                t2Idx.set( ri, 0 );
                            } else if ( t1Idx.shape( ri ) < t2Idx.shape( ri ) ) {
                                t1Idx.set( ri, 0 );
                                t2Idx.set( ri, t0Idx.get( ri ) );
                            }
                            ri++;
                        } else incrementing = false;
                    } else ri++;
                }
            }//setInto value in drn:
            t0_value[ t0Idx.i() ] = value;
            //increment on drain:
            t0Idx.increment();
            //NDConfiguration.Utility.increment(t0Idx, t0Shp);
            i++;
        }

    }

    private static void _deConvolve(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            Operation.TertiaryNDIConsumer operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        double[] t0_value = t0_drn.getDataAs( double[].class );

        // Incrementing if 'i>0' so that all indexes match:
        for ( int ii = 0; ii < i; ii++ ) {
            int ri = 0;
            while ( ri < rank ) {
                if ( t2Idx.get( ri ) == t2Idx.shape( ri ) ) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, 0 );
                } else {
                    t1Idx.set(
                            ri ,
                            t0Idx.shape( ri ) > t1Idx.shape( ri )
                                ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                : (t0Idx.get( ri ) + t2Idx.get( ri ))
                    );
                }
                ri++;
            }
        }

        // Looping through given range :
        while ( i < end ) {//increment on drain accordingly:
            int ri=0;
            while ( ri < rank ) {
                if ( t2Idx.get( ri ) == t2Idx.shape( ri ) ) {//setting 0
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
            while ( running ) {
                ri = ( ri == rank ? 0 : ri );
                if ( !incrementing ) {// := testing for match and applying operation:
                    boolean isMatch = true;
                    for ( int rii = 0; rii < rank; rii++ ) {
                        isMatch = (t1Idx.get( rii ) < t1Idx.shape( rii ) && t1Idx.get( rii ) >= 0) && isMatch;
                    }
                    value += (isMatch) ? operation.execute( t0Idx, t1Idx, t2Idx ) : 0;
                    incrementing = true;
                    ri = 0;
                } else { // incrementing:
                    if ( t2Idx.get( ri ) < t2Idx.shape( ri ) ) {
                        t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                        if ( t2Idx.get( ri ) == t2Idx.shape( ri ) ) {
                            running = (ri != rank - 1);
                            t1Idx.set( ri, t0Idx.get( ri ) );
                            t2Idx.set( ri, 0 );
                            ri++;
                        } else {
                            t1Idx.set(
                                    ri,
                                    t0Idx.shape( ri ) > t1Idx.shape( ri )
                                        ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                        : (t0Idx.get( ri ) + t2Idx.get( ri ))
                            );
                            incrementing = false;
                        }
                    } else ri++;
                }
            }
            //set value in drn:
            t0_value[ t0Idx.i() ] = value;
            //increment on drain:
            t0Idx.increment();
            //NDConfiguration.Utility.increment(t0Idx, t0Shp);
            i++;
        }
    }

}
