package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.operations.other.Reshape;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.internal.RecursiveExecutor;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

public class Broadcast extends AbstractFunctionalAlgorithm<Broadcast>
{
    public Broadcast( RecursiveExecutor finalExecutor )
    {
        super("broadcast");
        setIsSuitableFor(
            call->
            {
                boolean isInvalid =
                            !call.validate()
                                .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                                .isValid();

                if ( isInvalid )
                    return SuitabilityPredicate.UNSUITABLE;

                int maxRank = 0;
                for ( Tsr<?> t : call.getTensors() )
                    if ( t != null && t.rank() > maxRank ) maxRank = t.rank();

                for ( int i = 0; i < maxRank; i++ )
                {
                    int currentDim = -1;
                    for( Tsr<?> t : call.getTensors() )
                    {
                        if ( t != null && i < t.rank() ) {
                            if ( currentDim == -1 ) currentDim = t.shape( i );
                            else if ( currentDim != t.shape( i ) && currentDim != 1 && t.shape( i ) != 1 ) return 0.0f;
                        }
                    }
                }
                return SuitabilityPredicate.GOOD;
            }
        );
        setCanPerformForwardADFor( call -> {
            Tsr<?> last = null;
            for ( Tsr<?> t : call.getTensors() ) {
                if ( last != null && !last.shape().equals(t.shape()) ) return false;
                last = t;
            }
            return true;
        });
        setExecutionDispatcher(
            ( caller, call ) -> {
                int offset = ( call.getTsrOfType( Number.class, 0 ) == null ) ? 1 : 0;
                if (
                    call.getTsrOfType( Number.class, offset).shape().size() != call.getTsrOfType( Number.class, 1+offset).shape().size()
                ) // Creating a new tensor:
                {
                    Tsr<?>[] inputs = {call.getTsrOfType( Number.class, offset), call.getTsrOfType( Number.class, 1+offset) };
                    Reshape.makeFit( inputs, caller.isDoingAD() );
                    inputs = new Tsr[]{ null, inputs[0], inputs[1] };
                    CalcUtil.recursiveExecution( call.withTensors( inputs ), (executionCall, executor) -> null );
                    return inputs[0];
                }
                return CalcUtil.executeFor(caller, call, finalExecutor );
            }
        );
        setCallPreparation(
            call -> {
                Tsr<?>[] inputs = call.getTensors();
                Device device = call.getDevice();
                if ( inputs[ 0 ] == null ) // Creating a new tensor:
                {
                    int[] s1 = inputs[1].getNDConf().shape();
                    int[] s2 = inputs[2].getNDConf().shape();

                    assert s1.length == s2.length;
                    int[] outShape = new int[s1.length];

                    for ( int i = 0; i < outShape.length; i++ )
                        assert s1[ i ] == 1 || s2[ i ] == 1 || s1[ i ] == s2[ i ];

                    for ( int i = 0; i < outShape.length; i++ )
                        outShape[ i ] = ( s1[ i ] == 1 ? s2[ i ] : s1[ i ] );

                    Class<Object> type = (Class<Object>) inputs[ 1 ].getValueClass();
                    Tsr<?> output = Tsr.of(type).withShape(outShape).all( 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    try {
                        device.store( output );
                    } catch( Exception e ) {
                        e.printStackTrace();
                    }
                    inputs[ 0 ] = output;
                }
                return call;
            }
        );
    }

    public static WithForward<String> implementationForGPU( String postfix ) {
        return
            forward ->
                backward ->
                    CLImplementation.compiler()
                        .arity( 3 )
                        .kernelSource( Neureka.get().utility().readResource("kernels/broadcast_template.cl") )
                        .activationSource( forward )
                        .differentiationSource( backward )
                        .kernelPostfix( postfix )
                        .execution(
                            call -> {
                                int offset = ( call.getTsrOfType( Number.class, 0 ) != null ) ? 0 : 1;
                                int gwz = ( call.getTsrOfType( Number.class, 0 ) != null ) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                call.getDevice().getKernel(call)
                                        .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                        .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                        .passAllOf( call.getTsrOfType( Number.class, offset + 2 ) )
                                        .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                        .pass( call.getValOf( Arg.DerivIdx.class ) )
                                        .call( gwz );
                            }
                        )
                        .build();
    }

    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation(
                3,
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
            Fun.F64F64ToF64 operation = pairs.get( Fun.F64F64ToF64.class ).get( d );
            workload = (i, end) -> _broadcastF64( t0_drn, t1_src, t2_src, d, i, end, operation );
        }
        else if ( typeClass == Float.class ) {
            Fun.F32F32ToF32 operation = pairs.get( Fun.F32F32ToF32.class ).get( d );
            workload = (i, end) -> _broadcastF32( t0_drn, t1_src, t2_src, d, i, end, operation );
        }

        if ( workload == null )
            throw new IllegalArgumentException("");
        else
            return workload;
    }

    @Contract(pure = true)
    private static void _broadcastF64(
            Tsr<Number> t0_drn, Tsr<Number> t1_src, Tsr<Number> t2_src,
            int d, int i, int end,
            Fun.F64F64ToF64 operation
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
        double[] t0_value = t0_drn.getDataAs( double[].class );
        double[] t1_value = t1_src.getDataAs( double[].class );
        double[] t2_value = t2_src.getDataAs( double[].class );

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


    @Contract(pure = true)
    private static void _broadcastF32(
            Tsr<Number> t0_drn, Tsr<Number> t1_src, Tsr<Number> t2_src,
            int d, int i, int end,
            Fun.F32F32ToF32 operation
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
        float[] t0_value = t0_drn.getDataAs( float[].class );
        float[] t1_value = t1_src.getDataAs( float[].class );
        float[] t2_value = t2_src.getDataAs( float[].class );

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
