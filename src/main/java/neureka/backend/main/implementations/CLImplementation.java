package neureka.backend.main.implementations;


import neureka.backend.api.ImplementationFor;
import neureka.backend.api.template.implementations.AbstractImplementationFor;
import neureka.devices.opencl.KernelSource;
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

    public static SourceBuilder fromSource() {
        return new SourceBuilder();
    }

    public static ParsedCLImplementation.Builder compiler() {
        return new ParsedCLImplementation.Builder();
    }

    public static AdHocCompiler adHoc( KernelSource kernelProvider ) {
        return new AdHocCompiler( kernelProvider );
    }

    /**
     *  This builder builds the most basic type of {@link CLImplementation} which
     *  is in essence merely a wrapper for a lambda and the arity of this implementation.
     */
    public static class SourceBuilder
    {
        private ImplementationFor<OpenCLDevice> lambda;
        private int arity;
        private String kernelName;
        private String kernelSource;

        private SourceBuilder() { }

        /**
         * @param lambda The code which passes the call data to OpenCL and calls the kernel.
         * @return This builder instance to allow for method chaining.
         */
        public SourceBuilder lambda(ImplementationFor<OpenCLDevice> lambda) { this.lambda = lambda;return this; }
        public SourceBuilder arity(int arity) { this.arity = arity; return this; }
        public SourceBuilder kernelName(String kernelName) { this.kernelName = kernelName;return this; }
        public SourceBuilder kernelSource(String kernelSource) { this.kernelSource = kernelSource;return this; }
        public CLImplementation build() { return new SimpleCLImplementation(lambda, arity, kernelName, kernelSource); }
    }

    public static class AdHocCompiler {

        private final KernelSource _source;
        private int _arity;

        AdHocCompiler( KernelSource source ) { _source = source; }

        public AdHocCompiler arity( int arity ) { _arity = arity; return this; }

        public CLImplementation caller( ImplementationFor<OpenCLDevice> lambda) {
            return new AdHocClImplementation( lambda, _arity, _source );
        }

    }
}
