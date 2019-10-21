package neureka.core.device;

import neureka.core.Tsr;

public interface IDevice {

    void dispose();

    IDevice get(Tsr tensor);

    IDevice add(Tsr tensor);

    IDevice rmv(Tsr tensor);

    IDevice overwrite(Tsr tensor, double[] value);

    IDevice swap(Tsr former, Tsr replacement);

    IDevice execute(Tsr[] tsrs, int f_id, int d);

    double[] valueOf(Tsr tensor, boolean grd);

    float[] floatValueOf(Tsr tensor, boolean grd);

}
