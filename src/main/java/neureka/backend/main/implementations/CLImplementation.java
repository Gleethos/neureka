package neureka.backend.main.implementations;


import neureka.backend.api.ImplementationFor;
import neureka.backend.api.template.implementations.AbstractImplementationFor;
import neureka.devices.opencl.OpenCLDevice;
import neureka.devices.opencl.StaticKernelSource;

/**
 * This class is the ExecutorFor &lt; OpenCLDevice &gt; implementation
 * used to properly call an OpenCLDevice instance via the
 * ExecutionOn &lt; OpenCLDevice &gt; lambda implementation
 * receiving an instance of the ExecutionCall class.
*/
public abstract class CLImplementation extends AbstractImplementationFor<OpenCLDevice> implements StaticKernelSource
{
    protected CLImplementation(
            ImplementationFor<OpenCLDevice> execution,
            int arity
    ) {
        super( execution, arity );
    }
}
