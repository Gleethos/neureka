package neureka.backend.main.implementations.elementwise;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.opencl.OpenCLDevice;

public class CLRandomization implements ImplementationFor<OpenCLDevice> {
    @Override
    public Tsr<?> run(ExecutionCall<OpenCLDevice> call) {
        throw new IllegalStateException("Not yet implemented");
    }
}
