package neureka;

import neureka.acceleration.Device;

public class Neureka {

    static Device findAcceleratorByName(String name){
        return Device.find(name);
    }

    static String version(){
        return "0.0.0";
    }


    public static class settings
    {
        public static class debug
        {
            /**
             * Every derivative is calculated with respect to some graph node.
             * Graph nodes contain payload tensors.
             * A tensor might not always be used for backpropagation.
             * Therefore it will be deleted if possible.
             * Targeted tensors are either leave tensors (They require gradients)
             * or they are angle points between forward- and reverse-mode-AD!
             * In this case:
             * If the tensor is not needed for backpropagation it will be deleted.
             * The graph node will dereference the tensor either way.
             *
             * The flag determines this behaivior with respect to target nodes.
             * It is used for the test suit to validate that the right tensors were calculated.
             * This flag should not be modified in production! (memory leak)
             */
            public static boolean KEEP_DERIVATIVE_TARGET_PAYLOADS = false;



        }






    }

}
