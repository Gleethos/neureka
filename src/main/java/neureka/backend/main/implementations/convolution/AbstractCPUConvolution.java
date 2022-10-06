package neureka.backend.main.implementations.convolution;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUBiFun;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public abstract class AbstractCPUConvolution implements ImplementationFor<CPU>
{
    protected abstract CPUBiFun _getFun();

    @Override
    public Tsr<?> run( ExecutionCall<CPU> call )
    {
        SimpleCPUConvolution simpleConvolution = new SimpleCPUConvolution(call.input(1), call.input(2), call.input(0));

        if ( simpleConvolution.isSuitable() && call.getValOf(Arg.DerivIdx.class) < 0 )
            simpleConvolution.run();
        else
            _doNDConvolutionFor( call ); // General purpose ND convolution, -> any dimensionality.

        return call.input(0);
    }

    private void _doNDConvolutionFor( ExecutionCall<CPU> call )
    {
        call.getDevice()
            .getExecutor()
            .threaded(
                call.input(0).size(),
                _workloadFor(call)
            );
    }

    private CPU.RangeWorkload _workloadFor(
        ExecutionCall<CPU> call
    ) {
        Tsr<Number> t0_drn = call.input( Number.class, 0 );
        Tsr<Number> t1_src = call.input( Number.class, 1 );
        Tsr<Number> t2_src = call.input( Number.class, 2 );

        Class<?> typeClass = t0_drn.getItemType();

        int d = call.getDerivativeIndex();
        CPUBiFun f = _getFun();
        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class ) {
            if ( d < 0 )
                workload = (i, end) -> _convolve64( t0_drn, t1_src, t2_src, i, end, f );
            else
                workload = (i, end) -> _deConvolve64( t0_drn, t1_src, t2_src, i, end, f );
        }
        else if ( typeClass == Float.class ) {
            if ( d < 0 )
                workload = (i, end) -> _convolve32(t0_drn, t1_src, t2_src, i, end, f);
            else
                workload = (i, end) -> _deConvolve32( t0_drn, t1_src, t2_src, i, end, f );
        }

        if ( workload == null )
            throw new IllegalArgumentException("Could not create convolution worker for type class '"+typeClass+"'!");
        else
            return workload;
    }

    private static void _convolve64(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            CPUBiFun operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.indicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        double[] t0_value = t0_drn.getMut().getDataForWriting( double[].class );
        double[] t1_value = t1_src.getMut().getDataAs( double[].class );
        double[] t2_value = t2_src.getMut().getDataAs( double[].class );

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
                            running = ( ri != rank - 1 );
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
            }
            //set value in drn:
            t0_value[ t0Idx.i() ] = value;
            //increment on drain:
            t0Idx.increment();
            i++;
        }

    }


    private static void _deConvolve64(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            CPUBiFun operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.indicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        double[] t0_value = t0_drn.getMut().getDataForWriting( double[].class );
        double[] t1_value = t1_src.getMut().getDataAs( double[].class );
        double[] t2_value = t2_src.getMut().getDataAs( double[].class );

        assert t0_value != null;
        assert t1_value != null;
        assert t2_value != null;

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
            int ri = 0;
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
                            running = ( ri != rank - 1 );
                            t1Idx.set( ri, t0Idx.get( ri ) );
                            t2Idx.set( ri, 0 );
                            ri++;
                        } else {
                            t1Idx.set( ri,
                                    t0Idx.shape( ri ) > t1Idx.shape( ri )
                                            ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                            : (t0Idx.get( ri ) + t2Idx.get( ri ))
                            );
                            incrementing = false;
                        }
                    } else ri++;
                }
            }
            // set value in drn:
            t0_value[ t0Idx.i() ] = value;
            // increment on drain:
            t0Idx.increment();
            i++;
        }
    }

    // ---


    private static void _convolve32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            CPUBiFun operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.indicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        float[] t0_value = t0_drn.getMut().getDataForWriting( float[].class );
        float[] t1_value = t1_src.getMut().getDataAs( float[].class );
        float[] t2_value = t2_src.getMut().getDataAs( float[].class );

        while ( i < end )
        { // increment on drain accordingly:
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
                ri = ( ri == rank ? 0 : ri );
                if ( !incrementing ) {
                    value += operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] );
                    incrementing = true;
                    ri = 0;
                } else { // incrementing:
                    if ( t1Idx.get( ri ) < t1Idx.shape( ri ) && t2Idx.get( ri ) < t2Idx.shape( ri ) ) {
                        t1Idx.set( ri, t1Idx.get( ri ) + 1 );
                        t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                        if ( t1Idx.get( ri ) == t1Idx.shape( ri ) || t2Idx.get( ri ) == t2Idx.shape( ri )) {
                            running = ( ri != rank - 1 );
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
            }// set value in drain:
            t0_value[ t0Idx.i() ] = value;
            // increment on drain:
            t0Idx.increment();
            i++;
        }

    }


    private static void _deConvolve32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int i, int end,
            CPUBiFun operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.indicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        int rank = t0Idx.rank();

        float[] t0_value = t0_drn.getMut().getDataForWriting( float[].class );
        float[] t1_value = t1_src.getMut().getDataAs( float[].class );
        float[] t2_value = t2_src.getMut().getDataAs( float[].class );

        // Incrementing if 'i>0' so that all indexes match:
        for ( int ii = 0; ii < i; ii++ ) {
            int ri = 0;
            while ( ri < rank ) {
                if ( t2Idx.get( ri ) == t2Idx.shape( ri ) ) {
                    t1Idx.set( ri, t0Idx.get( ri ) );
                    t2Idx.set( ri, 0 );
                }
                else
                    t1Idx.set( ri ,
                            t0Idx.shape( ri ) > t1Idx.shape( ri )
                                    ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                    : (t0Idx.get( ri ) + t2Idx.get( ri ))
                    );
                ri++;
            }
        }

        // Looping through given range :
        while ( i < end ) { // increment on drain accordingly:
            int ri = 0;
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
                        isMatch = ( t1Idx.get( rii ) < t1Idx.shape( rii ) && t1Idx.get( rii ) >= 0 ) && isMatch;

                    value += ( isMatch ? operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] ) : 0 );
                    incrementing = true;
                    ri = 0;
                } else { // incrementing:
                    if ( t2Idx.get( ri ) < t2Idx.shape( ri ) ) {
                        t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                        if ( t2Idx.get( ri ) == t2Idx.shape( ri ) ) {
                            running = ( ri != rank - 1 );
                            t1Idx.set( ri, t0Idx.get( ri ) );
                            t2Idx.set( ri, 0 );
                            ri++;
                        } else {
                            t1Idx.set( ri,
                                    t0Idx.shape( ri ) > t1Idx.shape( ri )
                                            ? (t0Idx.get( ri ) - t2Idx.get( ri ))
                                            : (t0Idx.get( ri ) + t2Idx.get( ri ))
                            );
                            incrementing = false;
                        }
                    } else ri++;
                }
            }
            // set value in drain:
            t0_value[ t0Idx.i() ] = value;
            // increment on drain:
            t0Idx.increment();
            i++;
        }
    }

}
