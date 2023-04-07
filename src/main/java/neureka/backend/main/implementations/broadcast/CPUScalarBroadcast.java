package neureka.backend.main.implementations.broadcast;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUBiFun;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public abstract class CPUScalarBroadcast implements ImplementationFor<CPU>
{
    protected abstract CPUBiFun _getFun();
    protected abstract CPUBiFun _getDeriveAt0();
    protected abstract CPUBiFun _getDeriveAt1();

    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        call.getDevice()
            .getExecutor()
            .threaded(
                call.input(0).size(),
                _workloadFor(call)
            );

        return call.input(0);
    }

    public CPU.RangeWorkload _workloadFor(
            ExecutionCall<CPU> call
    ) {
        int offset = ( call.arity() == 3 ? 1 : 0 );
        Tsr<?> t0_drn = call.input( 0 );
        Tsr<?> src    = call.input( offset );

        Class<?> typeClass = call.input( 1 ).getItemType();

        int d = call.getDerivativeIndex();
        CPUBiFun f = ( d ==  0 ? _getDeriveAt0() : ( d == 1 ? _getDeriveAt1() : _getFun() ) );

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class ) {
            double value = call.input(Number.class, 1 + offset).at(0).get().doubleValue();
            double[] t0_value = t0_drn.mut().getDataForWriting(double[].class);
            double[] t1_value = src.mut().getDataAs(double[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while ( i < end ) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }
        if ( typeClass == Float.class ) {
            float value = call.input(Number.class, 1 + offset).at(0).get().floatValue();
            float[] t0_value = t0_drn.mut().getDataForWriting(float[].class);
            float[] t1_value = src.mut().getDataAs(float[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }
        if ( typeClass == Integer.class ) {
            int value = call.input(Number.class, 1 + offset).at(0).get().intValue();
            int[] t0_value = t0_drn.mut().getDataForWriting(int[].class);
            int[] t1_value = src.mut().getDataAs(int[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }
        if ( typeClass == Long.class ) {
            long value = call.input(Number.class, 1 + offset).at(0).get().longValue();
            long[] t0_value = t0_drn.mut().getDataForWriting(long[].class);
            long[] t1_value = src.mut().getDataAs(long[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }
        if ( typeClass == Short.class ) {
            short value = call.input(Number.class, 1 + offset).at(0).get().shortValue();
            short[] t0_value = t0_drn.mut().getDataForWriting(short[].class);
            short[] t1_value = src.mut().getDataAs(short[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }
        if ( typeClass == Byte.class ) {
            byte value = call.input(Number.class, 1 + offset).at(0).get().byteValue();
            byte[] t0_value = t0_drn.mut().getDataForWriting(byte[].class);
            byte[] t1_value = src.mut().getDataAs(byte[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }
        if ( typeClass == Character.class ) {
            char value = call.input(Character.class, 1 + offset).at(0).get();
            char[] t0_value = t0_drn.mut().getDataForWriting(char[].class);
            char[] t1_value = src.mut().getDataAs(char[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }
        if ( typeClass == Boolean.class ) {
            boolean value = call.input(Boolean.class, 1 + offset).at(0).get();
            boolean[] t0_value = t0_drn.mut().getDataForWriting(boolean[].class);
            boolean[] t1_value = src.mut().getDataAs(boolean[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }
        if ( t0_drn.mut().getData().getRef().getClass() == Object[].class ) {
            Object value = call.input( 1 + offset ).at(0).get();
            Object[] t0_value = t0_drn.mut().getDataForWriting(Object[].class);
            Object[] t1_value = src.mut().getDataAs(Object[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = f.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    i++;
                }
            };
        }

        if ( workload == null )
            throw new IllegalArgumentException("Unsupported type: " + typeClass);
        else
            return workload;
    }


}
