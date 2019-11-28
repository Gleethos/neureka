package neureka;

import neureka.acceleration.Device;

public class Neureka
{
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

        public static class ad
        {
            /**
             * After backward passes the used derivatives are usually not needed.
             * For debugging purposes however this flag remains and will
             * not allow for garbage collection of the used derivatives.
             */
            public static boolean RETAIN_GRAPH_DERIVATIVES_AFTER_BACKWARD = false;

            /**
             * This fla enables an optimization technique which only applies
             * gradients as soon as they are needed by a tensor (the tensor is used again).
             * If the flag is set to true
             * then error values will accumulate whenever it makes sense.
             * This technique however uses more memory but will
             * improve performance for some networks substantially.
             * The technique is termed JIT-Propagation.
             */
            public static boolean RETAIN_PENDING_ERROR_FOR_JITPROP = true;

            /**
             * Gradients will automatically be applied to tensors as soon as
             * they are being used for calculation.
             * This feature works well with JIT-Propagation.
             */
            public static boolean APPLY_GRADIENT_WHEN_TENSOR_IS_USED = false;

        }


    }

}
