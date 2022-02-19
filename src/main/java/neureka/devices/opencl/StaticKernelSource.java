package neureka.devices.opencl;

@FunctionalInterface
public interface StaticKernelSource extends KernelSource {

    default KernelCode getKernelCode() {
        return getKernelFor( null );
    }

}
