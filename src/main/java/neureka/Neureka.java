package neureka;

import neureka.acceleration.Device;

public class Neureka {

    static Device findAcceleratorByName(String name){
        return Device.find(name);
    }

    static String version(){
        return "0.0.0";
    }


}
