package neureka.core.device;

import neureka.core.Tsr;

public interface IDevice
{
    void dispose();

    IDevice get(Tsr tensor);

    IDevice add(Tsr tensor);

    boolean has(Tsr tensor);

    IDevice rmv(Tsr tensor);

    IDevice overwrite64(Tsr tensor, double[] value);

    IDevice overwrite32(Tsr tensor, float[] value);

    IDevice swap(Tsr former, Tsr replacement);

    IDevice execute(Tsr[] tsrs, int f_id, int d);

    double[] value64Of(Tsr tensor, boolean grd);

    float[] value32Of(Tsr tensor, boolean grd);
}
