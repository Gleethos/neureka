package neureka.acceleration;

import neureka.Tsr;
import neureka.acceleration.opencl.OpenCLPlatform;

import java.util.Collection;

public interface Device
{
    static Device find(String name){
        //TODO: Device plugin finding!
        Device[] result = {null};
        String search = name.toLowerCase();
        OpenCLPlatform.PLATFORMS().forEach((p)->{
            p.getDevices().forEach((d)->{
                String str = (d.name()+" | "+d.vendor()+" | "+d.type()).toLowerCase();
                if(str.contains(search)){
                    result[0] = d;
                }
            });
        });
        return result[0];
    }


    void dispose();

    Device get(Tsr tensor);

    Device add(Tsr tensor);

    Device add(Tsr tensor, Tsr parent);

    boolean has(Tsr tensor);

    Device rmv(Tsr tensor);

    Device overwrite64(Tsr tensor, double[] value);

    Device overwrite32(Tsr tensor, float[] value);

    Device swap(Tsr former, Tsr replacement);

    Device execute(Tsr[] tsrs, int f_id, int d);

    double[] value64Of(Tsr tensor);

    float[] value32Of(Tsr tensor);

    Collection<Tsr> tensors();

}
