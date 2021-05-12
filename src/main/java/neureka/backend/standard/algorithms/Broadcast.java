package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.Operation;
import neureka.devices.Device;
import neureka.dtype.NumericType;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

public class Broadcast extends AbstractFunctionalAlgorithm< Broadcast >
{

    public Broadcast() {
        super("broadcast");
        setIsSuitableFor(
                call->
                {
                    if (
                            !call.validate()
                            .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                            .isValid()
                    ) return 0.0f;

                    int maxRank = 0;
                    for ( Tsr t : call.getTensors() ) if ( t != null && t.rank() > maxRank ) maxRank = t.rank();
                    for ( int i = 0; i < maxRank; i++ )
                    {
                        int currentDim = -1;
                        for( Tsr t : call.getTensors() )
                        {
                            if ( t!=null && i < t.rank() ) {
                                if ( currentDim == -1 ) currentDim = t.shape( i );
                                else if ( currentDim != t.shape( i ) && currentDim != 1 && t.shape( i ) != 1 ) return 0.0f;
                            }
                        }
                    }
                    return 1.0f;
                }
        );
        setHandleInsteadOfDevice(
                ( caller, call ) -> {
                    int offset = ( call.getTsrOfType( Number.class, 0 ) == null ) ? 1 : 0;
                    if (
                            call.getTsrOfType( Number.class, 0+offset ).shape().size() != call.getTsrOfType( Number.class, 1+offset).shape().size()
                    ) // Creating a new tensor:
                    {
                        Tsr[] tsrs = {call.getTsrOfType( Number.class, 0+offset ), call.getTsrOfType( Number.class, 1+offset) };
                        Tsr.makeFit(tsrs, caller.isDoingAD() );
                        tsrs = new Tsr[]{null, tsrs[0], tsrs[1]};
                        call.getDevice().execute( call.withTensors( tsrs ) );
                        return tsrs[0];
                    }
                    return null;
                }
        );
        setInstantiateNewTensorsForExecutionIn(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                    {
                        int[] s1 = tsrs[1].getNDConf().shape();
                        int[] s2 = tsrs[2].getNDConf().shape();

                        assert s1.length == s2.length;
                        int[] newShape = new int[s1.length];

                        for ( int i = 0; i < newShape.length; i++ )
                            assert s1[ i ] == 1 || s2[ i ] == 1 || s1[ i ] == s2[ i ];

                        for ( int i = 0; i < newShape.length; i++ )
                            newShape[ i ] = ( s1[ i ] == 1 ) ? s2[ i ] : s1[ i ];

                        Tsr output = new Tsr( newShape, 0.0 );
                        output.setIsVirtual( false );
                        try {
                            device.store( output );
                        } catch( Exception e ) {
                            e.printStackTrace();
                        }
                        tsrs[ 0 ] = output;
                    }
                    return call;
                }
        );
    }

    public String getKernelSource() {
        return Neureka.instance().utility().readResource("kernels/broadcast_template.cl");
    }

    @Contract(pure = true)
    public static void broadcast(
            Tsr<Number> t0_drn, Tsr<Number> t1_src, Tsr<Number> t2_src,
            int d, int i, int end,
            Operation.TertiaryNDIConsumer operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = t1_src.getNDConf().shape();
        int[] t2Shp = (t2_src != null) ? t2_src.getNDConf().shape() : t1Shp;
        int rank = t0Shp.length;
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        t1Idx.set( t0_drn.IndicesOfIndex( i ) );
        NDIterator t2Idx = NDIterator.of( t2_src );
        double[] t0_value = t0_drn.value64();
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
                //setInto _value in drn:
                t0_value[t0Idx.i()] = operation.execute( t0Idx, t1Idx, t2Idx );
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
                    if (t0Shp[ri] == t1Shp[ri]) {
                        t1Idx.set( ri, t0Idx.get( ri ) );//all shapes are equal -> shape index can be inherited from origin!
                        t2Idx.set( ri, t0Idx.get( ri ) );
                    } else if (t0Shp[ri] > t1Shp[ri]) {
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
                    ri = ( ri == rank ) ? 0 : ri;
                    if ( !incrementing ) {
                        value += operation.execute( t0Idx, t1Idx, t2Idx );
                        incrementing = true;
                        ri = 0;
                    } else {//incrementing:
                        if ( t0Shp[ri] < t1Shp[ri] ) {//Only if origin shape is smaller than handle and drain!
                            t1Idx.set( ri, t1Idx.get( ri ) + 1 );
                            t2Idx.set( ri, t2Idx.get( ri ) + 1 );
                            if (t1Idx.get( ri ) == t1Shp[ri]) {
                                t1Idx.set( ri, 0 );
                                t2Idx.set( ri, 0 );
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
                t0_value[t0Idx.i()] = value;
                //increment on drain:
                t0Idx.increment();
                //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                i++;
            }
        }
    }



    @Contract(pure = true)
    public static void broadcast(
            Tsr t0_drn, Tsr t1_src, Tsr t2_src,
            int d, int i, int end,
            Operation.TertiaryNDAConsumer operation
    ) {
        NDConfiguration ndc0 = t0_drn.getNDConf();
        NDConfiguration ndc1 = t1_src.getNDConf();
        int[] t0Shp = ndc0.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int[] t1Shp = ndc1.shape();
        int[] t2Shp = (t2_src != null) ? t2_src.getNDConf().shape() : t1Shp;
        int rank = t0Shp.length;
        int[] t0Idx = ndc0.indicesOfIndex( i );
        int[] t1Idx = new int[ rank ];
        int[] t2Idx = new int[ rank ];
        double[] t0_value = (double[]) t0_drn.getData();
        if ( d < 0 ) {
            while ( i < end ) {//increment on drain accordingly:
                int ri = 0;
                while ( ri < rank ) {
                    if ( t1Shp[ri] == t2Shp[ri] ) {//Equal shapes -> out index is t1 & t2 index!for this ri
                        t1Idx[ri] = t0Idx[ri];
                        t2Idx[ri] = t0Idx[ri];
                    } else if ( t1Shp[ri] > t2Shp[ri] ) {//Current shape axis of t2 must be 1 !
                        t1Idx[ri] = t0Idx[ri];
                        t2Idx[ri] = 0;//...therefore it can be set to 0!
                    } else if ( t1Shp[ri] < t2Shp[ri] ) {//same principle:
                        t1Idx[ri] = 0;
                        t2Idx[ri] = t0Idx[ri];
                    }
                    ri++;
                }
                //----------
                //setInto _value in drn:
                t0_value[ndc0.indexOfIndices(t0Idx)] = operation.execute( t0Idx, t1Idx, t2Idx );
                //increment on drain:
                NDConfiguration.Utility.increment(t0Idx, t0Shp);
                i++;
            }
        }
        else//---//Note: src2 is now former drain!
        {
            while ( i < end ) {//increment on drain accordingly:
                int ri = 0;
                while ( ri < rank ) {
                    if (t0Shp[ri] == t1Shp[ri]) {
                        t1Idx[ri] = t0Idx[ri];//all shapes are equal -> shape index can be inherited from origin!
                        t2Idx[ri] = t0Idx[ri];
                    } else if ( t0Shp[ri] > t1Shp[ri] ) {
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
                while ( running ) {
                    ri = ( ri == rank ) ? 0 : ri;
                    if ( !incrementing ) {
                        value += operation.execute( t0Idx, t1Idx, t2Idx );
                        incrementing = true;
                        ri = 0;
                    } else {//incrementing:
                        if ( t0Shp[ri] < t1Shp[ri] ) {//Only if origin shape is smaller than handle and drain!
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
                t0_value[ndc0.indexOfIndices(t0Idx)] = value;
                //increment on drain:
                NDConfiguration.Utility.increment(t0Idx, t0Shp);
                i++;
            }
        }
    }



}
