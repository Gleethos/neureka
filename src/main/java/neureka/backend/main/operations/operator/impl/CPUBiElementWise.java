package neureka.backend.main.operations.operator.impl;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.main.algorithms.Functions;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.algorithms.internal.FunTuple;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public class CPUBiElementWise
{

    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation( -1, CPUBiElementWise::_newWorkloadFor );
    }

    private static CPU.RangeWorkload _newWorkloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> pairs
    ) {
        FunTuple<Fun.F64F64ToF64> funF64 = pairs.get(Fun.F64F64ToF64.class);
        FunTuple<Fun.F32F32ToF32> funF32 = pairs.get(Fun.F32F32ToF32.class);
        FunTuple<Fun.I32I32ToI32> funI32 = pairs.get(Fun.I32I32ToI32.class);
        Class<?> typeClass = call.input( 1 ).getItemType();

        int d = call.getDerivativeIndex();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class )
            workload = _newWorkloadF64(  call.input( 0 ), call.input( 1 ), call.input( 2 ), funF64.get(d) );

        if ( typeClass == Float.class )
            workload = _newWorkloadF32(  call.input( 0 ), call.input( 1 ), call.input( 2 ), funF32.get(d) );

        if ( typeClass == Integer.class )
            workload = _newWorkloadI32(  call.input( 0 ), call.input( 1 ), call.input( 2 ), funI32.get(d) );

        if ( workload == null )
            throw new IllegalArgumentException("");
        else
            return workload;
    }

    private static CPU.RangeWorkload _newWorkloadF64(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            Fun.F64F64ToF64 operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );
        double[] t0_val = t0_drn.getUnsafe().getDataForWriting( double[].class );
        double[] t1_val = t1_src.getUnsafe().getDataAs( double[].class );
        double[] t2_val = t2_src.getUnsafe().getDataAs( double[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() )
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        else {
            if ( isSimple )
                return (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_val[i] = operation.invoke(t1_val[i], t2_val[i]);
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    NDIterator t2Idx = NDIterator.of(t2_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t1_src.indicesOfIndex(i));
                    t2Idx.set(t2_src.indicesOfIndex(i));
                    while ( i < end ) {//increment on drain accordingly:
                        //setInto _value in drn:
                        t0_val[t0Idx.i()] = operation.invoke(t1_val[t1Idx.i()], t2_val[t2Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        t2Idx.increment();
                        i++;
                    }
                };
        }
    }

    private static CPU.RangeWorkload _newWorkloadF32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            Fun.F32F32ToF32 operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );

        float[] t0_val = t0_drn.getUnsafe().getDataForWriting( float[].class );
        float[] t1_val = t1_src.getUnsafe().getDataAs( float[].class );
        float[] t2_val = t2_src.getUnsafe().getDataAs( float[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() )
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        else {
            if ( isSimple )
                return  (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_val[i] = operation.invoke(t1_val[i], t2_val[i]);
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    NDIterator t2Idx = NDIterator.of(t2_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t1_src.indicesOfIndex(i));
                    t2Idx.set(t2_src.indicesOfIndex(i));
                    while ( i < end ) {//increment on drain accordingly:
                        //setInto _value in drn:
                        t0_val[t0Idx.i()] = operation.invoke(t1_val[t1Idx.i()], t2_val[t2Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        t2Idx.increment();
                        i++;
                    }
                };
        }
    }

    private static CPU.RangeWorkload _newWorkloadI32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            Fun.I32I32ToI32 operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );
        int[] t0_val = (int[]) t0_drn.getUnsafe().getData().getRef();
        int[] t1_val = t1_src.getUnsafe().getDataAs( int[].class );
        int[] t2_val = t2_src.getUnsafe().getDataAs( int[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() )
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        else {
            if ( isSimple )
                return  (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_val[i] = operation.invoke(t1_val[i], t2_val[i]);
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    NDIterator t2Idx = NDIterator.of(t2_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t1_src.indicesOfIndex(i));
                    t2Idx.set(t2_src.indicesOfIndex(i));
                    while ( i < end ) {//increment on drain accordingly:
                        //setInto _value in drn:
                        t0_val[t0Idx.i()] = operation.invoke(t1_val[t1Idx.i()], t2_val[t2Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        t2Idx.increment();
                        i++;
                    }
                };
        }
    }

    public static class FunTriple<T extends Fun> implements FunTuple<T> {

        private final T _a, _d1, _d2;

        public FunTriple(T a, T d, T d2 ) {
            _a = a; _d1 = d; _d2 = d2;
        }

        @Override
        public T get(int derivativeIndex) {
            return ( derivativeIndex < 0 ? _a : ( derivativeIndex == 0 ? _d1 : _d2 ) );
        }

        @Override
        public Class<T> getType() { return (Class<T>) _a.getClass(); }
    }

}
