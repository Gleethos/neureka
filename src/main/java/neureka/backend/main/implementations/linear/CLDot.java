package neureka.backend.main.implementations.linear;

import neureka.Shape;
import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.operations.linear.internal.opencl.CLSum;
import neureka.devices.opencl.KernelCaller;
import neureka.devices.opencl.OpenCLDevice;

import java.util.function.Supplier;

/**
 *  Performs a dot product on two vectors using OpenCL.
 */
public class CLDot implements ImplementationFor<OpenCLDevice>
{
    @Override
    public Tensor<?> run(ExecutionCall<OpenCLDevice> call ) {
        // First we unpack the input tensors:
        Tensor<Float> c = call.input(Float.class, 0);
        Tensor<Float> a = call.input(Float.class, 1);
        Tensor<Float> b = call.input(Float.class, 2);
        OpenCLDevice device = call.getDevice();

        if ( a.rank() != 1 || b.rank() != 1 )
            throw new IllegalArgumentException("Input tensors must be vectors.");

        int size = a.shape(0);
        if ( b.shape(0) != size )
            throw new IllegalArgumentException("Input vectors must have the same length.");

        // First we multiply the two vectors:
        String kernelName = "multiply_arrays_for_dot_product";
        Supplier<String> code = () ->
                    "__kernel void " + kernelName + "(__global const float* a, \n" +
                    "                              __global const float* b, \n" +
                    "                              __global float* c,\n" +
                    "                              const int n) {\n" +
                    "    int i = get_global_id(0);\n" +
                    "    if (i < n) {\n" +
                    "        c[i] = a[i] * b[i];\n" +
                    "    }\n" +
                    "}";

        Tensor<Float> temp = Tensor.of(Float.class, Shape.of(size), 0).to(device).mut().setIsVirtual(false);

        // Kernels are cached, so if it is already compiled, it will be retrieved from the cache:
        KernelCaller caller = device.findOrCompileAdHocKernel(kernelName, code);
        // We call OpenCL to do the work:
        caller.pass(a).pass(b).pass(temp).pass(size).call(new long[]{size}, null);

        Tensor<Float> out = CLSum.run(temp, device);
        c.mut().at(0).set(out.item());
        return c;
    }
}