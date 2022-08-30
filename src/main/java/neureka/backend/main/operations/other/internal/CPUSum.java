package neureka.backend.main.operations.other.internal;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.host.CPU;

/**
 *  An implementation of the sum and may algorithm running on the CPU.
 *  This algorithm splits the provided input tensor into chucks which
 *  are then reduced to local sum values.
 *  This happens iteratively until only a single value is left.
 */
public class CPUSum implements ImplementationFor<CPU>
{
    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        if ( call.getDevice() != CPU.get() )
            throw new IllegalArgumentException("This implementation is only available for the CPU!");
        Tsr<?> in = call.input(0) == null ? call.input(1) : call.input(0);
        in.setIsVirtual(false);
        return _runRecursively(in, CPU.get());
    }

    private Tsr<?> _runRecursively(Tsr<?> in, CPU device)
    {
        CPU.JVMExecutor executor = device.getExecutor();
        int RTS = 128; // Register tile size
        final int SIZE = in.size();

        double fraction = (double) SIZE / (double) RTS;
        // Determining optimal number of tiles!
        int N;
        // Check if fraction is an integer
        if ( fraction == Math.floor(fraction) )
            N = (int) fraction;
        else
            N = (int) Math.ceil(fraction); // The last tile we do a partial reduction (bound check)

        if ( in.size() == 1 )  return in;

        Class<?> type = in.itemType();

        if ( type == Float.class ) {
            float[] inData = in.getUnsafe().getData().getRef(float[].class);
            float[] out = new float[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                float value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tsr<Float> reduced = Tsr.of(Float.class, new int[]{N}, out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Double.class ) {
            double[] inData = in.getUnsafe().getData().getRef(double[].class);
            double[] out = new double[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                double value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tsr<Double> reduced = Tsr.of(Double.class, new int[]{N}, out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Integer.class ) {
            int[] inData = in.getUnsafe().getData().getRef(int[].class);
            int[] out = new int[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                int value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tsr<Integer> reduced = Tsr.of(Integer.class, new int[]{N}, out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Long.class ) {
            long[] inData = in.getUnsafe().getData().getRef(long[].class);
            long[] out = new long[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                long value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tsr<Long> reduced = Tsr.of(Long.class, new int[]{N}, out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Short.class ) {
            short[] inData = in.getUnsafe().getData().getRef(short[].class);
            short[] out = new short[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                short value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tsr<Short> reduced = Tsr.of(Short.class, new int[]{N}, out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Byte.class ) {
            byte[] inData = in.getUnsafe().getData().getRef(byte[].class);
            byte[] out = new byte[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                byte value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tsr<Byte> reduced = Tsr.of(Byte.class, new int[]{N}, out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( Number.class.isAssignableFrom(type) ) {
            Object[] inData = in.getUnsafe().getData().getRef(Object[].class);
            Number[] out = new Number[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                Number value = (Number) inData[offset];
                offset++;
                for ( int i = offset; i < limit; ++i ) value = value.doubleValue() + ((Number)inData[i]).doubleValue();
                out[ni] = value;
            });
            Tsr<Number> reduced = Tsr.of(Number.class, new int[]{N}, out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else
            throw new IllegalArgumentException("Unsupported type: " + type);
    }

}
