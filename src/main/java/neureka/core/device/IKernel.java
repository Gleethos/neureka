package neureka.core.device;

public interface IKernel {


    double[] values();
    int[] shapes();
    int[] translations();
    int[] pointers();
    int[] idx();

    double[] value();
    int[] shape();
    int[] translation();
}
