package neureka;

import neureka.acceleration.Device;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;

public class Neureka
{
    public static Device findAcceleratorByName(String name){
        return Device.find(name);
    }

    public static Function create(String expression){
        return create(expression, true);
    }

    public static Function create(String expression, boolean doAD){
        return Function.create(expression, doAD);
    }


    public static String version(){
        return "1.0.0";
    }

    public static class Settings
    {

        public static boolean isLocked(){
            return  _isLocked;
        }

        public static void setIsLocked(boolean locked){
            _isLocked = locked;
        }

        private static boolean _isLocked = false;

        public static void reset(){
            Debug.reset();
            AD.reset();
            Indexing.reset();
        }

        public static class Debug
        {
            public static void reset(){
                _keepDerivativeTargetPayloads = false;
            }
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
            private static boolean _keepDerivativeTargetPayloads = false;

            public static boolean keepDerivativeTargetPayloads(){
                return _keepDerivativeTargetPayloads;
            }

            public static void setKeepDerivativeTargetPayloads(boolean keep){
                if(_isLocked) return;
                _keepDerivativeTargetPayloads = keep;
            }


        }

        public static class AD // Auto-Differentiation
        {
            public static void reset(){
                _retainGraphDerivativesAfterBackward = false;
                _retainPendingErrorForJITProp = true;
                _applyGradientWhenTensorIsUsed = false;
            }
            /**
             * After backward passes the used size are usually not needed.
             * For debugging purposes however this flag remains and will
             * not allow for garbage collection of the used size.
             */
            private static boolean _retainGraphDerivativesAfterBackward = false;

            public static boolean retainGraphDerivativesAfterBackward(){
                return _retainGraphDerivativesAfterBackward;
            }
            public static void setRetainGraphDerivativesAfterBackward(boolean retain){
                if(_isLocked) return;
                _retainGraphDerivativesAfterBackward = retain;
            }

            /**
             * This fla enables an optimization technique which only applies
             * gradients as soon as they are needed by a tensor (the tensor is used again).
             * If the flag is set to true
             * then error values will accumulate whenever it makes sense.
             * This technique however uses more memory but will
             * improve performance for some networks substantially.
             * The technique is termed JIT-Propagation.
             */
            private static boolean _retainPendingErrorForJITProp = true;

            public static boolean retainPendingErrorForJITProp(){
                return _retainPendingErrorForJITProp;
            }

            public static void setRetainPendingErrorForJITProp(boolean retain){
                if(_isLocked) return;
                _retainPendingErrorForJITProp = retain;
            }

            /**
             * Gradients will automatically be applied to tensors as soon as
             * they are being used for calculation.
             * This feature works well with JIT-Propagation.
             */
            private static boolean _applyGradientWhenTensorIsUsed = false;

            public static boolean applyGradientWhenTensorIsUsed(){
                return _applyGradientWhenTensorIsUsed;
            }

            public static void setApplyGradientWhenTensorIsUsed(boolean apply){
                if(_isLocked) return;
                _applyGradientWhenTensorIsUsed = apply;
            }

        }

        public static class Indexing
        {
            public static void reset(){
                _legacyIndexing = false;
            }

            private static boolean _legacyIndexing = false;


            public static boolean legacy(){
                return _legacyIndexing;
            }

            public static void setLegacy(boolean enabled){
                if(_isLocked) return;
                _legacyIndexing = enabled;//NOTE: gpu code must recompiled! (OpenCLPlatform)
            }

        }

    }

}
