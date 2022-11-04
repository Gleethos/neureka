package neureka.devices.opencl;

public interface StaticKernelSource extends KernelSource {

    KernelCode[] getKernelCode();

}
