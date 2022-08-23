package neureka.backend.main.operations.linear.internal.opencl;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.opencl.KernelCaller;
import neureka.devices.opencl.OpenCLDevice;

import java.util.function.Supplier;

public class Reduce implements ImplementationFor<OpenCLDevice>
{
    public enum Type { MIN, MAX }

    private final Type _type;
    private final static int RTS = 64; // Register Tile Size
    private final String _comparator;

    public Reduce(Type type) {
        String comparator = "";
        switch (type) {
            case MIN: comparator = "current < value"; break;
            case MAX: comparator = "current > value"; break;
            default: throw new IllegalArgumentException("Unsupported reduction type: "+type);
        }
        _comparator = comparator;
        _type = type;
    }

    @Override
    public Tsr<?> run(ExecutionCall<OpenCLDevice> call) {
        Tsr<Float> in = call.input(0) == null ? call.input(Float.class, 1) : call.input(Float.class, 0);
        int index = _runRecursively(in, call.getDevice());
        return Tsr.of(Integer.class, new int[]{1}, index);
    }

    private int _runRecursively(Tsr<Float> in, OpenCLDevice device)
    {
        final int SIZE = in.size();

        double fraction = (double) SIZE / (double) RTS;
        // Determining optimal number of tiles!
        int N;
        // Check if fraction is an integer
        if ( fraction == Math.floor(fraction) )
            N = (int) fraction;
        else
            N = (int) Math.ceil(fraction); // The last tile we do a partial reduction (bound check)

        Tsr<Integer> out = Tsr.of(Integer.class, new int[]{N}, 0).to(device);
        out.setIsVirtual(false);

        if ( in.size() == 1 ) {
            assert out.size() == 1;
            return out.at(0).get();
        }

        String kernelName = "fast_"+_type.name().toLowerCase()+"_reduce_RTS"+RTS;

        Supplier<String> code = () ->
                        "   #define RTS "+RTS+"                                                                             \n" +
                        "   __kernel void "+kernelName+"(                                                                   \n" +
                        "               const int size,                                                                     \n" +
                        "               const __global float* in,                                                           \n" +
                        "                     __global int* out, // indices                                                 \n" +
                        "   ) {                                                                                             \n" +
                        "       size_t ni = get_global_id(0); //   global N-tile id                                         \n" +
                        "                                                                                                   \n" +
                        "       uint offset = ni * RTS;                                                                     \n" +
                        "       uint limit = min( offset + RTS, size );                                                     \n" +
                        "       float value = in[offset];                                                                   \n" +
                        "       int found_index = offset;                                                                   \n" +
                        "       offset++;                                                                                   \n" +
                        "                                                                                                   \n" +
                        "       #pragma unroll                                                                              \n" +
                        "       for ( uint i=offset; i < limit; ++i ) {                                                     \n" +
                        "           float current = in[i];                                                                  \n" +
                        "           if ( "+ _comparator +" ) {                                                                 \n" +
                        "               value = current;                                                                    \n" +
                        "               found_index = i;                                                                    \n" +
                        "           }                                                                                       \n" +
                        "       }                                                                                           \n" +
                        "       out[ni] = found_index;                                                                      \n" +
                        "   }                                                                                               \n";

        KernelCaller caller =
                device.hasAdHocKernel(kernelName)
                        ? device.getAdHocKernel(kernelName)
                        : device.compileAdHocKernel(kernelName, code.get()).getAdHocKernel(kernelName);

        long[] local =  null; // This kernel does not have local memory (uses register/private memory instead)
        long[] global = new long[]{ N };

        caller.pass(SIZE).pass( in ).pass( out ).call( global, local );

        if ( N > 1 ) {
            Tsr<Float> reduced = _fetch(in, out, device);
            out.getUnsafe().delete();
            return _runRecursively(reduced, device);
        } else
            return out.at(0).get();
    }

    /**
     *  Creates and return a new tensor with the size of the
     *  {@code indices} tensor but with th values targeted in the {@code in}
     *  argument.
     *  All of this is done on a simple index to array entry mapping kernel!
     */
    private Tsr<Float> _fetch(
            Tsr<Float> in, Tsr<Integer> indices, OpenCLDevice device
    ) {
        Tsr<Float> out = Tsr.of(Float.class, new int[]{indices.size()}, 0);

        String kernelName = "indices_to_values_mapper";

        Supplier<String> code = () ->
                    "   __kernel void " + kernelName + "(                         \n" +
                    "               const __global int* indices,                  \n" +
                    "               const __global float* in,                     \n" +
                    "                     __global float* out                     \n" +
                    "   ) {                                                       \n" +
                    "       size_t i = get_global_id(0);  //   global id          \n" +
                    "       out[i] = in[indices[i]];                              \n" +
                    "   }                                                         \n";

        KernelCaller caller =
                device.hasAdHocKernel(kernelName)
                        ? device.getAdHocKernel(kernelName)
                        : device.compileAdHocKernel(kernelName, code.get()).getAdHocKernel(kernelName);

        long[] local =  null; // This kernel does not have local memory (uses register/private memory instead)
        long[] global = new long[]{ indices.size() };

        caller.pass( indices ).pass( in ).pass( out ).call( global, local );
        return out;
    }

}


