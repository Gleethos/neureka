package neureka.backend.main.implementations.linear;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.opencl.KernelCaller;
import neureka.devices.opencl.OpenCLDevice;

import java.util.function.Supplier;

/**
 *  Performs a dot product on two vectors using OpenCL.
 */
public class CLDot implements ImplementationFor<OpenCLDevice> {
    @Override
    public Tsr<?> run( ExecutionCall<OpenCLDevice> call ) {
        // First we unpack the input tensors:
        Tsr<Float> c = call.input(Float.class, 0);
        Tsr<Float> a = call.input(Float.class, 1);
        Tsr<Float> b = call.input(Float.class, 2);

        if ( a.rank() != 1 || b.rank() != 1 )
            throw new IllegalArgumentException("Input tensors must be vectors.");

        int size = a.shape(0);
        if ( b.shape(0) != size )
            throw new IllegalArgumentException("Input vectors must have the same length.");

        String kernelName = "simple_dot_product";
        Supplier<String> code = () -> // TODO: Create faster kernels making use of local and private memory!!

                        "__kernel void dot_product(int n, __global float *a, __global float *b, __global float *output) {\n" +
                        "    __local float a_local[256];\n" +
                        "    __local float b_local[256];\n" +
                        "    int i = get_global_id(0);\n" +
                        "    int local_id = get_local_id(0);\n" +
                        "    float sum = 0.0f;\n" +
                        "    while (i < n) {\n" +
                        "        a_local[local_id] = a[i];\n" +
                        "        b_local[local_id] = b[i];\n" +
                        "        i += get_global_size(0);\n" +
                        "        local_id += get_local_size(0);\n" +
                        "    }\n" +
                        "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
                        "    for (int j = 0; j < get_local_size(0); j++) {\n" +
                        "        sum += a_local[j] * b_local[j];\n" +
                        "    }\n" +
                        "    __private float partial_sum = sum;\n" +
                        "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
                        "    if (local_id == 0) {\n" +
                        "        atomic_add(output, partial_sum);\n" +
                        "    }\n" +
                        "}";

        // Kernels are cached, so if it is already compiled, it will be retrieved from the cache:
        KernelCaller caller =
                    call.getDevice().hasAdHocKernel(kernelName)
                        ? call.getDevice().findAdHocKernel(kernelName).orElseThrow(()-> new RuntimeException("Kernel not found!"))
                        : call.getDevice().compileAndGetAdHocKernel(kernelName, code.get());

        // We call OpenCL to do the work:
        caller.pass(size).pass(a).pass(b).pass(c)
                .call(new long[]{size}, null);

        return c;
    }
    /*
        Here an alternative kernel code:
        Supplier<String> code = () ->
                            "#define VECTOR_SIZE 4\n" +
                            "__kernel void dot_product(int n, __global float4 *a, __global float4 *b, __global float *output) {\n" +
                            "    int i = get_global_id(0) * VECTOR_SIZE;\n" +
                            "    float4 sum = (float4)(0.0f);\n" +
                            "    while (i < n) {\n" +
                            "        sum += a[i/VECTOR_SIZE] * b[i/VECTOR_SIZE];\n" +
                            "        i += get_global_size(0) * VECTOR_SIZE;\n" +
                            "    }\n" +
                            "    float partial_sum = sum.x + sum.y + sum.z + sum.w;\n" +
                            "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
                            "    atomic_add(output, partial_sum);\n" +
                            "}";
     */

    /*
    String kernelName = "fast_dot_product";
    Supplier<String> code = () ->
        "__kernel void dot_product(__global const float* a, __global const float* b, __global float* c, const int n) {\n"
        + "    const int localSize = get_local_size(0);\n"
        + "    const int localId = get_local_id(0);\n"
        + "    const int globalId = get_global_id(0);\n"
        + "    __local float localSums[256];\n" // We assume a maximum work-group size of 256
        + "    float localSum = 0.0f;\n"
        + "    for (int i = globalId; i < n; i += get_global_size(0)) {\n"
        + "        localSum += a[i] * b[i];\n"
        + "    }\n"
        + "    localSums[localId] = localSum;\n"
        + "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        + "    for (int i = localSize / 2; i > 0; i /= 2) {\n"
        + "        if (localId < i) {\n"
        + "            localSums[localId] += localSums[localId + i];\n"
        + "        }\n"
        + "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        + "    }\n"
        + "    if (localId == 0) {\n"
        + "        atomic_add(&c[0], localSums[0]);\n"
        + "    }\n"
        + "}\n";
     */
}