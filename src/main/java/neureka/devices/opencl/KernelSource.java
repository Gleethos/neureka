package neureka.devices.opencl;

import neureka.backend.api.ExecutionCall;

/**
 *  Provides kernel source code for a provided {@link ExecutionCall}.
 */
@FunctionalInterface
public interface KernelSource {

    KernelCode getKernelFor( ExecutionCall<OpenCLDevice> call );

}
