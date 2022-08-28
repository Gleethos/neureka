package neureka.backend.main.operations.linear.internal.opencl;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.opencl.KernelCaller;
import neureka.devices.opencl.OpenCLDevice;

import java.util.function.Supplier;

public class CLSum implements ImplementationFor<OpenCLDevice>
{
    @Override
    public Tsr<?> run(ExecutionCall<OpenCLDevice> call) {
        return _partialSum(call.input(Float.class, 0), call.getDevice());
    }

    /**
     *  This method compiles and executes the kernel that will return the sum of the
     *  elements in the {@code in} tensor.
     */
    private static Tsr<Float> _partialSum(
        Tsr<Float> in, OpenCLDevice device
    ) {
        final long RTS = device.maxWorkGroupSize(); // Register tile size
        final int SIZE = in.size();
        long localSize = device.maxWorkGroupSize();
        while (SIZE % localSize != 0) { localSize--; } // We want to have a multiple of the max workgroup size + as large as possible
        final int N = (int) (SIZE / localSize); // The number of partial sums

        long[] local  = new long[]{ localSize };
        long[] global = new long[]{(long) SIZE };

        Tsr<Float> out;

        if ( localSize == 1 ) { // Oh, the user wants to process a prime number of elements... sigh! Ok let's do it (slower)!
            double fraction = (double) SIZE / (double) RTS;
            final int newN;
            // Determining optimal number of tiles!
            // Check if fraction is an integer
            if ( fraction == Math.floor(fraction) )
                newN = (int) fraction;
            else
                newN = (int) Math.ceil(fraction); // The last tile we do a partial reduction (bound check)

            out = Tsr.of(Float.class, new int[]{newN}, 0).to(device).setIsVirtual(false);
            KernelCaller caller = _processPrivate(RTS, device);
            caller.pass(SIZE).pass(in).pass(out).call(global, local);
        }
        else
        {
            out = Tsr.of(Float.class, new int[]{N}, 0).to(device).setIsVirtual(false);
            KernelCaller caller = _processLocal(device);
            caller.pass(in).pass(out).passLocalFloats((int) localSize).call(global, local);
        }

        if ( N > 1 ) {
            Tsr<Float> reduced = _partialSum(out, device);
            out.getUnsafe().delete();
            return reduced;
        }
        return out;
    }

    private static KernelCaller _processPrivate( long RTS, OpenCLDevice device )
    {
        String kernelName = "fast_private_sum_reduction_RTS"+RTS;
        Supplier<String> code = () ->
                        "   #define RTS "+RTS+"                                                                    \n" +
                        "   __kernel void "+kernelName+"(                                                          \n" +
                        "               const int size,                                                            \n" +
                        "               const __global float* in,                                                  \n" +
                        "                     __global float* out                                                  \n" +
                        "   ) {                                                                                    \n" +
                        "       size_t ni = get_global_id(0); //   global N-tile id                                \n" +
                        "                                                                                          \n" +
                        "       int offset = ni * RTS;                                                             \n" +
                        "       int limit = min( offset + RTS, size ); // Boundary condition!                      \n" +
                        "       float value = in[offset];                                                          \n" +
                        "       offset++;                                                                          \n" +
                        "                                                                                          \n" +
                        "       #pragma unroll                                                                     \n" +
                        "       for ( uint i=offset; i < limit; ++i )                                              \n" +
                        "           value += in[i];                                                                \n" +
                        "                                                                                          \n" +
                        "       out[ni] = value;                                                                   \n" +
                        "   }                                                                                      \n";
        return
            device.hasAdHocKernel(kernelName)
                    ? device.getAdHocKernel(kernelName)
                    : device.compileAdHocKernel(kernelName, code.get()).getAdHocKernel(kernelName);
    }

    private static KernelCaller _processLocal(
            OpenCLDevice device
    ) {
        String kernelName = "fast_local_mem_based_sum";
        Supplier<String> code = () ->
                "                                                                                                  \n" +
                "    int div(int i) { return i % 2 == 1 ? (i+1) / 2 : i / 2;  }                                    \n" +
                "                                                                                                  \n" +
                "    __kernel void "+kernelName+" (                                                                \n" +
                "       __global const float *input,                                                               \n" +
                "       __global float *partialSums,                                                               \n" +
                "       __local float *localSums                                                                   \n" +
                "    ){                                                                                            \n" +
                "        uint local_id = get_local_id(0);                                                          \n" +
                "        uint group_size = get_local_size(0);                                                      \n" +
                "                                                                                                  \n" +
                "        // Copy from global to local memory                                                       \n" +
                "        localSums[local_id] = input[get_global_id(0)];                                            \n" +
                "                                                                                                  \n" +
                "        // Loop for computing localSums : divide WorkGroup into 2 parts                           \n" +
                "        uint last = group_size;                                                                   \n" +
                "        for (uint stride = div(group_size); stride > 0 && last > stride; stride=div(stride))      \n" +
                "        {                                                                                         \n" +
                "             // Waiting for each 2x2 addition into given workgroup                                \n" +
                "             barrier(CLK_LOCAL_MEM_FENCE);                                                        \n" +
                "                                                                                                  \n" +
                "             // Add elements 2 by 2 between local_id and local_id + stride                        \n" +
                "             uint right_id = local_id + stride; // We copy from the right part.                   \n" +
                "             if (local_id < stride && right_id < last )                                           \n" +
                "                 localSums[local_id] += localSums[local_id + stride];                             \n" +
                "             last = stride;                                                                       \n" +
                "        }                                                                                         \n" +
                "                                                                                                  \n" +
                "        // Write result into partialSums[nWorkGroups]                                             \n" +
                "        if (local_id == 0)                                                                        \n" +
                "            partialSums[get_group_id(0)] = localSums[0];                                          \n" +
                "    }                                                                                             \n";

        return
                device.hasAdHocKernel(kernelName)
                        ? device.getAdHocKernel(kernelName)
                        : device.compileAdHocKernel(kernelName, code.get()).getAdHocKernel(kernelName);
    }

}
