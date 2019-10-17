package neureka.core.device;

import neureka.core.Tsr;

public interface IDevice {

    void dispose();

    IDevice add(Tsr tensor);

    IDevice rmv(Tsr tensor);

    IDevice overwrite(Tsr tensor, double[] value);

    IDevice swap(Tsr former, Tsr replacement);

}
