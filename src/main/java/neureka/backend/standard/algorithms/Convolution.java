package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

public class Convolution extends AbstractFunctionalAlgorithm<Convolution>
{
    public Convolution() {
        super("convolution");
        setIsSuitableFor(
            call ->
                call.validate()
                    .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                    .basicSuitability()
        );
    }


    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/convolution_template.cl");
    }


    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation(
                    (call, pairs) ->
                            call.getDevice()
                                .getExecutor()
                                .threaded(
                                        call.getTsrOfType( Number.class, 0 ).size(),
                                        _newWorkloadFor( call, pairs )
                                )
                );
    }


    private static CPU.RangeWorkload _newWorkloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> pairs
    ) {
        Tsr<Number> t0_drn = call.getTsrOfType( Number.class, 0 );
        Tsr<Number> t1_src = call.getTsrOfType( Number.class, 1 );
        Tsr<Number> t2_src = call.getTsrOfType( Number.class, 2 );

        Class<?> typeClass = t0_drn.getValueClass();

        int d = call.getDerivativeIndex();
        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class ) {
            Fun.F64F64ToF64 operation = pairs.get( Fun.F64F64ToF64.class ).get( -1 );
            if ( d < 0 )
                workload = (i, end) -> _convolve64( t0_drn, t1_src, t2_src, i, end, operation );
            else
                workload = (i, end) -> _deConvolve64( t0_drn, t1_src, t2_src, i, end, operation );
        }
        else if ( typeClass == Float.class ) {
            Fun.F32F32ToF32 operation = pairs.get( Fun.F32F32ToF32.class ).get( -1 );
            if ( d < 0 )
                workload = (i, end) -> _convolve32( t0_drn, t1_src, t2_src, i, end, operation );
            else
                workload = (i, end) -> _deConvolve32( t0_drn, t1_src, t2_src, i, end, operation );
        }

        if ( workload == null )
            throw new IllegalArgumentException("");
        else
            return workload;
    }


    @Contract(pure = true)
    private static void _convolve64(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            Fun.F64F64ToF64 operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        double[] t0_value = t0_drn.getDataAs( double[].class );
        double[] t1_value = t1_src.getDataAs( double[].class );
        double[] t2_value = t2_src.getDataAs( double[].class );

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
                    value += operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] );
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


    private static void _deConvolve64(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            Fun.F64F64ToF64 operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        double[] t0_value = t0_drn.getDataAs( double[].class );
        double[] t1_value = t1_src.getDataAs( double[].class );
        double[] t2_value = t2_src.getDataAs( double[].class );

        // Incrementing if 'i>0' so that all indexes match:
        for ( int ii = 0; ii < i; ii++ ) {
            int ri = 0;
            while ( ri < rank ) {
                if ( t2Idx.get( ri ) == t2Idx.shape( ri ) ) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, 0 );
                }
                else
                    t1Idx.set(
                            ri ,
                            t0Idx.shape( ri ) > t1Idx.shape( ri )
                                    ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                    : (t0Idx.get( ri ) + t2Idx.get( ri ))
                        );
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
                }
                else
                    t1Idx.set( ri, (t0Idx.shape( ri ) > t1Idx.shape( ri ))
                                    ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                    : (t0Idx.get( ri ) + t2Idx.get( ri ))
                            );
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
                    for ( int rii = 0; rii < rank; rii++ )
                        isMatch = (t1Idx.get( rii ) < t1Idx.shape( rii ) && t1Idx.get( rii ) >= 0) && isMatch;

                    value += (isMatch) ? operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] ) : 0;
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

    // ---

    @Contract(pure = true)
    private static void _convolve32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            Fun.F32F32ToF32 operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        float[] t0_value = t0_drn.getDataAs( float[].class );
        float[] t1_value = t1_src.getDataAs( float[].class );
        float[] t2_value = t2_src.getDataAs( float[].class );

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
            float value = 0;
            boolean running = true;
            boolean incrementing = false;
            while ( running ) {
                ri = ( ri == rank ) ? 0 : ri;
                if ( !incrementing ) {
                    value += operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] );
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


    private static void _deConvolve32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            Fun.F32F32ToF32 operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        float[] t0_value = t0_drn.getDataAs( float[].class );
        float[] t1_value = t1_src.getDataAs( float[].class );
        float[] t2_value = t2_src.getDataAs( float[].class );

        // Incrementing if 'i>0' so that all indexes match:
        for ( int ii = 0; ii < i; ii++ ) {
            int ri = 0;
            while ( ri < rank ) {
                if ( t2Idx.get( ri ) == t2Idx.shape( ri ) ) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, 0 );
                }
                else
                    t1Idx.set(
                            ri ,
                            t0Idx.shape( ri ) > t1Idx.shape( ri )
                                    ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                    : (t0Idx.get( ri ) + t2Idx.get( ri ))
                    );
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
                }
                else
                    t1Idx.set( ri, (t0Idx.shape( ri ) > t1Idx.shape( ri ))
                            ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                            : (t0Idx.get( ri ) + t2Idx.get( ri ))
                    );
                ri++;
            }
            //----------
            float value = 0;
            boolean running = true;
            boolean incrementing = false;
            while ( running ) {
                ri = ( ri == rank ? 0 : ri );
                if ( !incrementing ) {// := testing for match and applying operation:
                    boolean isMatch = true;
                    for ( int rii = 0; rii < rank; rii++ )
                        isMatch = (t1Idx.get( rii ) < t1Idx.shape( rii ) && t1Idx.get( rii ) >= 0) && isMatch;

                    value += (isMatch) ? operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] ) : 0;
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
