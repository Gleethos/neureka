package neureka.backend.main.operations.other.internal;

import neureka.Shape;
import neureka.Tensor;
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
    public Tensor<?> run(ExecutionCall<CPU> call ) {
        if ( call.getDevice() != CPU.get() )
            throw new IllegalArgumentException("This implementation is only available for the CPU!");
        Tensor<?> in = call.input(0) == null ? call.input(1) : call.input(0);
        in.mut().setIsVirtual(false);
        return _runRecursively(in, CPU.get());
    }

    private Tensor<?> _runRecursively(Tensor<?> in, CPU device)
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
            float[] inData = in.mut().getData().as(float[].class);
            float[] out = new float[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                float value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tensor<Float> reduced = Tensor.of(Float.class, Shape.of(N), out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Double.class ) {
            double[] inData = in.mut().getData().as(double[].class);
            double[] out = new double[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                double value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tensor<Double> reduced = Tensor.of(Double.class, Shape.of(N), out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Integer.class ) {
            int[] inData = in.mut().getData().as(int[].class);
            int[] out = new int[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                int value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tensor<Integer> reduced = Tensor.of(Integer.class, Shape.of(N), out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Long.class ) {
            long[] inData = in.mut().getData().as(long[].class);
            long[] out = new long[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                long value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tensor<Long> reduced = Tensor.of(Long.class, Shape.of(N), out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Short.class ) {
            short[] inData = in.mut().getData().as(short[].class);
            short[] out = new short[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                short value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tensor<Short> reduced = Tensor.of(Short.class, Shape.of(N), out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( type == Byte.class ) {
            byte[] inData = in.mut().getData().as(byte[].class);
            byte[] out = new byte[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                byte value = 0;
                for ( int i = offset; i < limit; ++i ) value += inData[i];
                out[ni] = value;
            });
            Tensor<Byte> reduced = Tensor.of(Byte.class, Shape.of(N), out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else if ( Number.class.isAssignableFrom(type) ) {
            Object[] inData = in.mut().getData().as(Object[].class);
            Number[] out = new Number[N];
            executor.threaded( N, ni -> {
                int offset = ni * RTS;
                int limit = Math.min( offset + RTS, SIZE );
                Number value = (Number) inData[offset];
                offset++;
                for ( int i = offset; i < limit; ++i ) value = value.doubleValue() + ((Number)inData[i]).doubleValue();
                out[ni] = value;
            });
            Tensor<Number> reduced = Tensor.of(Number.class, Shape.of(N), out);
            if ( N > 1 )
                return _runRecursively(reduced, device);
            else
                return reduced; // This is the final result!
        }
        else
            throw new IllegalArgumentException("Unsupported type: " + type);
    }

}
