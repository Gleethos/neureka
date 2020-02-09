package neureka.acceleration;

import neureka.acceleration.opencl.OpenCLPlatform;

import java.util.Collection;

public interface Device<Type>
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

    Device get(Type tensor);

    Device add(Type tensor);

    Device add(Type tensor, Type parent);

    boolean has(Type tensor);

    Device rmv(Type tensor);

    Device cleaning(Type tensor, Runnable action);

    Device overwrite64(Type tensor, double[] value);

    Device overwrite32(Type tensor, float[] value);

    Device swap(Type former, Type replacement);

    Device execute(Type[] Types, int f_id, int d);

    double[] value64Of(Type tensor);

    float[] value32Of(Type tensor);

    Collection<Type> tensors();




}
