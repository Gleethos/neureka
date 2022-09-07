package neureka.backend.main.operations.function.scalar;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.algorithms.Functions;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public class CPUElementwiseActivation implements ImplementationFor<CPU>
{
    private final ImplementationFor<CPU> _impl;

    public CPUElementwiseActivation(ScalarFun fun) {
        _impl = Functions.implementation( 1, CPUElementwiseActivation::_newWorkloadFor )
                .with(Fun.F64ToF64.pair(fun::activate, fun::derive))
                .with(Fun.F32ToF32.pair(fun::activate, fun::derive))
                .with(Fun.I32ToI32.pair(fun::activate, fun::derive))
                .with(Fun.I64ToI64.pair(fun::activate, fun::derive))
                .with(Fun.I8ToI8.pair(fun::activate, fun::derive))
                .with(Fun.I16ToI16.pair(fun::activate, fun::derive))
                .with(Fun.BoolToBool.pair(fun::activate, fun::derive))
                .with(Fun.CharToChar.pair(fun::activate, fun::derive))
                .with(Fun.ObjToObj.pair(fun::activate, fun::derive))
                .get();
    }

    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        return _impl.run(call);
    }

    private static CPU.RangeWorkload _newWorkloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> funs
    ) {
        Tsr<?> t0_drn = call.input( 0 );
        Tsr<?> t1_src = call.input( 1 );
        Class<?> typeClass = t0_drn.getItemType();
        Class<?> rightTypeClass = t1_src.getItemType();

        assert !t0_drn.isVirtual();
        assert !t1_src.isVirtual();

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple();

        int d = call.getDerivativeIndex();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class )
        {
            Fun.F64ToF64 fun = funs.get(Fun.F64ToF64.class).get(d);
            double[] t0_value = t0_drn.getUnsafe().getDataForWriting( double[].class );

            if ( rightTypeClass == Integer.class )
            {
                int[] t1_value = (int[]) t1_src.getUnsafe().getData().getRef();
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t0_drn.indicesOfIndex(i));
                    while (i < end) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
            }
            else
            {
                double[] t1_value = t1_src.getUnsafe().getDataAs(double[].class);
                if ( isSimple )
                    workload = (start, end) -> {
                        for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                    };
                else
                    workload = (i, end) -> {
                        NDIterator t0Idx = NDIterator.of(t0_drn);
                        NDIterator t1Idx = NDIterator.of(t1_src);
                        t0Idx.set(t0_drn.indicesOfIndex(i));
                        t1Idx.set(t0_drn.indicesOfIndex(i));
                        while (i < end) { // increment on drain accordingly:
                            //setInto _value in drn:
                            t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                            //increment on drain:
                            t0Idx.increment();
                            t1Idx.increment();
                            i++;
                        }
                    };
            }
        }
        else if ( typeClass == Float.class )
        {
            Fun.F32ToF32 fun = funs.get(Fun.F32ToF32.class).get(d);
            assert fun != null;
            float[] t0_value = t0_drn.getUnsafe().getDataForWriting( float[].class );
            float[] t1_value = t1_src.getUnsafe().getDataAs(float[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Integer.class )
        {
            Fun.I32ToI32 fun = funs.get(Fun.I32ToI32.class).get(d);
            assert fun != null;
            int[] t0_value = (int[]) t0_drn.getUnsafe().getData().getRef();
            int[] t1_value = t1_src.getUnsafe().getDataAs(int[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Long.class )
        {
            Fun.I64ToI64 fun = funs.get(Fun.I64ToI64.class).get(d);
            assert fun != null;
            long[] t0_value = (long[]) t0_drn.getUnsafe().getData().getRef();
            long[] t1_value = t1_src.getUnsafe().getDataAs(long[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Byte.class )
        {
            Fun.I8ToI8 fun = funs.get(Fun.I8ToI8.class).get(d);
            assert fun != null;
            byte[] t0_value = (byte[]) t0_drn.getUnsafe().getData().getRef();
            byte[] t1_value = t1_src.getUnsafe().getDataAs(byte[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Short.class )
        {
            Fun.I16ToI16 fun = funs.get(Fun.I16ToI16.class).get(d);
            assert fun != null;
            short[] t0_value = (short[]) t0_drn.getUnsafe().getData().getRef();
            short[] t1_value = t1_src.getUnsafe().getDataAs(short[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Boolean.class )
        {
            Fun.BoolToBool fun = funs.get(Fun.BoolToBool.class).get(d);
            assert fun != null;
            boolean[] t0_value = (boolean[]) t0_drn.getUnsafe().getData().getRef();
            boolean[] t1_value = t1_src.getUnsafe().getDataAs(boolean[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Character.class )
        {
            Fun.CharToChar fun = funs.get(Fun.CharToChar.class).get(d);
            assert fun != null;
            char[] t0_value = (char[]) t0_drn.getUnsafe().getData().getRef();
            char[] t1_value = t1_src.getUnsafe().getDataAs(char[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        } else {
            try {
                Fun.ObjToObj fun = funs.get(Fun.ObjToObj.class).get(d);
                assert fun != null;
                Object[] t0_value = (Object[]) t0_drn.getUnsafe().getData().getRef();
                Object[] t1_value = t1_src.getUnsafe().getDataAs(Object[].class);
                if (isSimple)
                    workload = (start, end) -> {
                        for (int i = start; i < end; i++) t0_value[i] = fun.invoke(t1_value[i]);
                    };
                else
                    workload = (i, end) -> {
                        NDIterator t0Idx = NDIterator.of(t0_drn);
                        NDIterator t1Idx = NDIterator.of(t1_src);
                        t0Idx.set(t0_drn.indicesOfIndex(i));
                        t1Idx.set(t0_drn.indicesOfIndex(i));
                        while (i < end) { // increment on drain accordingly:
                            //setInto _value in drn:
                            t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                            //increment on drain:
                            t0Idx.increment();
                            t1Idx.increment();
                            i++;
                        }
                    };
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        if ( workload == null )
            throw new IllegalArgumentException(
                    "Operand types '"+typeClass.getSimpleName()+"' and '"+rightTypeClass.getSimpleName()+"' not supported."
            );

        return workload;
    }

}
