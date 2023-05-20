package neureka.backend.main.implementations.elementwise;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.opencl.OpenCLDevice;

public class CLRandomization implements ImplementationFor<OpenCLDevice> {
    @Override
    public Tensor<?> run(ExecutionCall<OpenCLDevice> call) {
        throw new IllegalStateException("Not yet implemented");
    }
}
