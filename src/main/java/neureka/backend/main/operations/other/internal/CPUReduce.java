package neureka.backend.main.operations.other.internal;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.host.CPU;

/**
 *  An implementation of the min and max algorithm running on the CPU.
 *  This algorithm splits the provided input tensor into chucks which
 *  are then reduced to local min and max values.
 *  This happens iteratively until only a single value is left.
 *  Each workload also returns the index of the found min/max value,
 *  which is important for backpropagation...
 */
public class CPUReduce implements ImplementationFor<CPU>
{
    private interface ComparatorF32 { boolean compare(float current, float value); }
    private interface ComparatorF64 { boolean compare(double current, double value); }
    private interface ComparatorI32 { boolean compare(int current, int value); }
    private interface ComparatorI64 { boolean compare(long current, long value); }
    private interface ComparatorI8  { boolean compare(byte current, byte value); }
    private interface ComparatorI16 { boolean compare(short current, short value); }

    public enum Type {
        MIN, MAX;

        private ComparatorF32 getFloatComparator() {
            switch (this) {
                case MIN: return (current, value) -> current < value;
                case MAX: return (current, value) -> current > value;
                default: throw new IllegalArgumentException("Unsupported reduction type: "+this);
            }
        }
        private ComparatorF64 getDoubleComparator() {
            switch (this) {
                case MIN: return (current, value) -> current < value;
                case MAX: return (current, value) -> current > value;
                default: throw new IllegalArgumentException("Unsupported reduction type: "+this);
            }
        }
        private ComparatorI32 getIntComparator() {
            switch (this) {
                case MIN: return (current, value) -> current < value;
                case MAX: return (current, value) -> current > value;
                default: throw new IllegalArgumentException("Unsupported reduction type: "+this);
            }
        }
        private ComparatorI64 getLongComparator() {
            switch (this) {
                case MIN: return (current, value) -> current < value;
                case MAX: return (current, value) -> current > value;
                default: throw new IllegalArgumentException("Unsupported reduction type: "+this);
            }
        }
        private ComparatorI8 getByteComparator() {
            switch (this) {
                case MIN: return (current, value) -> current < value;
                case MAX: return (current, value) -> current > value;
                default: throw new IllegalArgumentException("Unsupported reduction type: "+this);
            }
        }
        private ComparatorI16 getShortComparator() {
            switch (this) {
                case MIN: return (current, value) -> current < value;
                case MAX: return (current, value) -> current > value;
                default: throw new IllegalArgumentException("Unsupported reduction type: "+this);
            }
        }
    }

    private final Type _type;


    public CPUReduce(Type type) {
        _type = type;
    }


    @Override
    public Tsr<Integer> run(ExecutionCall<CPU> call) {
        if ( call.getDevice() != CPU.get() )
            throw new IllegalArgumentException("This implementation is only available for the CPU!");
        Tsr<?> in = call.input(0) == null ? call.input(1) : call.input(0);
        int index = _runRecursively(in, CPU.get());
        return Tsr.of(Integer.class, new int[]{1}, index);
    }

    private int _runRecursively(Tsr<?> in, CPU device)
    {
        CPU.JVMExecutor executor = device.getExecutor();
        int RTS = 64;
        final int SIZE = in.size();

        double fraction = (double) SIZE / (double) RTS;
        // Determining optimal number of tiles!
        int N;
        // Check if fraction is an integer
        if ( fraction == Math.floor(fraction) )
            N = (int) fraction;
        else
            N = (int) Math.ceil(fraction); // The last tile we do a partial reduction (bound check)

        int[] out = new int[N];

        if ( in.size() == 1 ) {
            assert out.length == 1;
            return out[0];
        }
        Class<?> type = in.itemType();

        if ( type == Float.class ) {
            ComparatorF32 comparator = _type.getFloatComparator();
            float[] inData = in.getUnsafe().getDataForWriting(float[].class);
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                float value = inData[offset];
                int found_index = offset;
                offset++;
                for ( int i=offset; i < limit; ++i ) {
                    float current = inData[i];
                    if ( comparator.compare(current, value) ) {
                        value = current; found_index = i;
                    }
                }
                out[ni] = found_index;
            });
            if ( N > 1 ) {
                float[] reduced = new float[out.length];
                executor.threaded( out.length, (start, end) -> { for ( int i=start; i < end; ++i ) reduced[i] = inData[out[i]];});
                return out[_runRecursively(Tsr.of(Float.class, new int[]{out.length}, reduced), device)];
            }
        }
        if ( type == Double.class ) {
            ComparatorF64 comparator = _type.getDoubleComparator();
            double[] inData = in.getUnsafe().getDataForWriting(double[].class);
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                double value = inData[offset];
                int found_index = offset;
                offset++;
                for ( int i=offset; i < limit; ++i ) {
                    double current = inData[i];
                    if ( comparator.compare(current, value) ) {
                        value = current; found_index = i;
                    }
                }
                out[ni] = found_index;
            });
            if ( N > 1 ) {
                double[] reduced = new double[out.length];
                executor.threaded( out.length, (start, end) -> { for ( int i=start; i < end; ++i ) reduced[i] = inData[out[i]];});
                return out[_runRecursively(Tsr.of(Double.class, new int[]{out.length}, reduced), device)];
            }
        }
        if ( type == Integer.class ) {
            ComparatorI32 comparator = _type.getIntComparator();
            int[] inData = in.getUnsafe().getDataForWriting(int[].class);
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                int value = inData[offset];
                int found_index = offset;
                offset++;
                for ( int i=offset; i < limit; ++i ) {
                    int current = inData[i];
                    if ( comparator.compare(current, value) ) {
                        value = current; found_index = i;
                    }
                }
                out[ni] = found_index;
            });
            if ( N > 1 ) {
                int[] reduced = new int[out.length];
                executor.threaded( out.length, (start, end) -> { for ( int i=start; i < end; ++i ) reduced[i] = inData[out[i]];});
                return out[_runRecursively(Tsr.of(Integer.class, new int[]{out.length}, reduced), device)];
            }
        }
        if ( type == Long.class ) {
            ComparatorI64 comparator = _type.getLongComparator();
            long[] inData = in.getUnsafe().getDataForWriting(long[].class);
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                long value = inData[offset];
                int found_index = offset;
                offset++;
                for ( int i=offset; i < limit; ++i ) {
                    long current = inData[i];
                    if ( comparator.compare(current, value) ) {
                        value = current; found_index = i;
                    }
                }
                out[ni] = found_index;
            });
            if ( N > 1 ) {
                long[] reduced = new long[out.length];
                executor.threaded( out.length, (start, end) -> { for ( int i=start; i < end; ++i ) reduced[i] = inData[out[i]];});
                return out[_runRecursively(Tsr.of(Long.class, new int[]{out.length}, reduced), device)];
            }
        }
        if ( type == Short.class ) {
            ComparatorI16 comparator = _type.getShortComparator();
            short[] inData = in.getUnsafe().getDataForWriting(short[].class);
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                short value = inData[offset];
                int found_index = offset;
                offset++;
                for ( int i=offset; i < limit; ++i ) {
                    short current = inData[i];
                    if ( comparator.compare(current, value) ) {
                        value = current; found_index = i;
                    }
                }
                out[ni] = found_index;
            });
            if ( N > 1 ) {
                short[] reduced = new short[out.length];
                executor.threaded( out.length, (start, end) -> { for ( int i=start; i < end; ++i ) reduced[i] = inData[out[i]];});
                return out[_runRecursively(Tsr.of(Short.class, new int[]{out.length}, reduced), device)];
            }
        }
        if ( type == Byte.class ) {
            ComparatorI8 comparator = _type.getByteComparator();
            byte[] inData = in.getUnsafe().getDataForWriting(byte[].class);
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                byte value = inData[offset];
                int found_index = offset;
                offset++;
                for ( int i=offset; i < limit; ++i ) {
                    byte current = inData[i];
                    if ( comparator.compare(current, value) ) {
                        value = current; found_index = i;
                    }
                }
                out[ni] = found_index;
            });
            if ( N > 1 ) {
                byte[] reduced = new byte[out.length];
                executor.threaded( out.length, (start, end) -> { for ( int i=start; i < end; ++i ) reduced[i] = inData[out[i]];});
                return out[_runRecursively(Tsr.of(Byte.class, new int[]{out.length}, reduced), device)];
            }
        }

        return out[0];
    }

}
