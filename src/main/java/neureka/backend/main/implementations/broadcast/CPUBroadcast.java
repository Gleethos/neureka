package neureka.backend.main.implementations.broadcast;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUBiFun;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public abstract class CPUBroadcast implements ImplementationFor<CPU>
{
    protected CPUBroadcast() {}

    protected abstract CPUBiFun _getFun();
    protected abstract CPUBiFun _getDeriveAt0();
    protected abstract CPUBiFun _getDeriveAt1();

    @Override
    public Tsr<?> run( ExecutionCall<CPU> call ) {
        call.getDevice()
                .getExecutor()
                .threaded(
                    call.input(0).size(),
                    _newWorkloadFor(call)
                );

        return call.input(0);
    }

    private CPU.RangeWorkload _newWorkloadFor(
            ExecutionCall<CPU> call
    ) {
        Tsr<Number> t0_drn = call.input( Number.class, 0 );
        Tsr<Number> t1_src = call.input( Number.class, 1 );
        Tsr<Number> t2_src = call.input( Number.class, 2 );

        t0_drn.mut().setIsVirtual(false);

        Class<?> typeClass = t0_drn.getItemType();

        int d = call.getDerivativeIndex();
        CPUBiFun f = ( d ==  0 ? _getDeriveAt0() : ( d == 1 ? _getDeriveAt1() : _getFun() ) );

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class ) {
            workload = (i, end) -> _broadcastF64( t0_drn, t1_src, t2_src, d, i, end, f );
        }
        else if ( typeClass == Float.class ) {
            workload = (i, end) -> _broadcastF32( t0_drn, t1_src, t2_src, d, i, end, f );
        }

        if ( workload == null )
            throw new IllegalArgumentException(
                    "Failed to find an implementation for tensor with type '"+typeClass.getSimpleName()+"'!"
                );
        else
            return workload;
    }


    private static void _broadcastF64(
            Tsr<Number> t0_drn, Tsr<Number> t1_src, Tsr<Number> t2_src,
            int d, int i, int end,
            CPUBiFun operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = t1_src.getNDConf().shape();
        int[] t2Shp = (t2_src != null) ? t2_src.getNDConf().shape() : t1Shp;
        int rank = t0Shp.length;
        assert t2_src != null;
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src, NDIterator.NonVirtual.TRUE );
        t0Idx.set( t0_drn.indicesOfIndex( i ) );
        t1Idx.set( t0_drn.indicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src, NDIterator.NonVirtual.TRUE );
        double[] t0_value = t0_drn.mut().getDataForWriting( double[].class );
        double[] t1_value = t1_src.mut().getDataAs( double[].class );
        double[] t2_value = t2_src.mut().getDataAs( double[].class );

        if ( d < 0 ) {
            while ( i < end ) {//increment on drain accordingly:
                int ri = 0;
                while ( ri < rank ) {
                    if ( t1Shp[ri] == t2Shp[ri] ) {//Equal shapes -> out index is t1 & t2 index!for this ri
                        t1Idx.set( ri, t0Idx.get( ri ) );
                        t2Idx.set( ri, t0Idx.get( ri ) );
                    } else if ( t1Shp[ri] > t2Shp[ri] ) {//Current shape axis of t2 must be 1 !
                        t1Idx.set( ri, t0Idx.get( ri ) );
                        t2Idx.set( ri, 0 );//...therefore it can be set to 0!
                    } else if ( t1Shp[ri] < t2Shp[ri] ) {//same principle:
                        t1Idx.set( ri, 0 );
                        t2Idx.set( ri, t0Idx.get( ri ) );
                    }
                    ri++;
                }
                //----------
                //set in value in drn:
                t0_value[t0Idx.i()] = operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] );
                //increment on drain:
                t0Idx.increment();
                //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                i++;
            }
        }
        else //---// Note: src2 is now former drain!
        {
            while ( i < end ) {//increment on drain accordingly:
                int ri = 0;
                while ( ri < rank ) {
                    if ( t0Shp[ri] == t1Shp[ri] ) {
                        t1Idx.set( ri, t0Idx.get( ri ) );//all shapes are equal -> shape index can be inherited from origin!
                        t2Idx.set( ri, t0Idx.get( ri ) );
                        if ( t2Shp[ri] == 1 ) t2Idx.set( ri, 0 );
                        else t2Idx.set( ri, t0Idx.get( ri ) );
                    } else if ( t0Shp[ri] > t1Shp[ri] ) {
                        t1Idx.set( ri, 0 );//Current origin index is larger: index can be inherited!
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
                    ri = ( ri == rank ? 0 : ri );
                    if ( !incrementing ) {
                        value += operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] );
                        incrementing = true;
                        ri = 0;
                    } else {//incrementing:
                        if ( t0Shp[ri] < t1Shp[ri] ) {//Only if origin shape is smaller than handle and drain!
                            t1Idx.set( ri, t1Idx.get( ri ) + 1 );
                            t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                            if ( t1Idx.get( ri ) == t1Shp[ri] ) {
                                t1Idx.set( ri, 0 );
                                t2Idx.set( ri, 0 );
                                running = (ri != rank - 1);
                                ri++;
                            }
                            else incrementing = false;//return to calculation!

                        } else {
                            running = (ri != rank - 1);
                            ri++;
                        }
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



    private static void _broadcastF32(
            Tsr<Number> t0_drn, Tsr<Number> t1_src, Tsr<Number> t2_src,
            int d, int i, int end,
            CPUBiFun operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = t1_src.getNDConf().shape();
        int[] t2Shp = (t2_src != null) ? t2_src.getNDConf().shape() : t1Shp;
        int rank = t0Shp.length;
        assert t2_src != null;
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src, NDIterator.NonVirtual.TRUE );
        t0Idx.set( t0_drn.indicesOfIndex( i ) );
        t1Idx.set( t0_drn.indicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src, NDIterator.NonVirtual.TRUE );
        float[] t0_value = t0_drn.mut().getDataForWriting( float[].class );
        float[] t1_value = t1_src.mut().getDataAs( float[].class );
        float[] t2_value = t2_src.mut().getDataAs( float[].class );

        if ( d < 0 ) {
            while ( i < end ) {//increment on drain accordingly:
                int ri = 0;
                while ( ri < rank ) {
                    if ( t1Shp[ri] == t2Shp[ri] ) {//Equal shapes -> out index is t1 & t2 index!for this ri
                        t1Idx.set( ri, t0Idx.get( ri ) );
                        t2Idx.set( ri, t0Idx.get( ri ) );
                    } else if ( t1Shp[ri] > t2Shp[ri] ) {//Current shape axis of t2 must be 1 !
                        t1Idx.set( ri, t0Idx.get( ri ) );
                        t2Idx.set( ri, 0 );//...therefore it can be set to 0!
                    } else if ( t1Shp[ri] < t2Shp[ri] ) {//same principle:
                        t1Idx.set( ri, 0 );
                        t2Idx.set( ri, t0Idx.get( ri ) );
                    }
                    ri++;
                }
                //----------
                //set in value in drn:
                t0_value[t0Idx.i()] = operation.invoke( t1_value[t1Idx.i()], t2_value[t2Idx.i()] );
                //increment on drain:
                t0Idx.increment();
                //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                i++;
            }
        }
        else //---// Note: src2 is now former drain!
        {
            while ( i < end ) {//increment on drain accordingly:
                int ri = 0;
                while ( ri < rank ) {
                    if ( t0Shp[ri] == t1Shp[ri] ) {
                        t1Idx.set( ri, t0Idx.get( ri ) );//all shapes are equal -> shape index can be inherited from origin!
                        t2Idx.set( ri, t0Idx.get( ri ) );
                        if ( t2Shp[ri] == 1 ) t2Idx.set( ri, 0 );
                        else t2Idx.set( ri, t0Idx.get( ri ) );
                    } else if ( t0Shp[ri] > t1Shp[ri] ) {
                        t1Idx.set( ri, 0 );//Current origin index is larger: index can be inherited!
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
                    } else {//incrementing:
                        if ( t0Shp[ri] < t1Shp[ri] ) {//Only if origin shape is smaller than handle and drain!
                            t1Idx.set( ri, t1Idx.get( ri ) + 1 );
                            t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                            if ( t1Idx.get( ri ) == t1Shp[ri] ) {
                                t1Idx.set( ri, 0 );
                                t2Idx.set( ri, 0 );
                                running = (ri != rank - 1);
                                ri++;
                            }
                            else incrementing = false;//return to calculation!

                        } else {
                            running = (ri != rank - 1);
                            ri++;
                        }
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


}
