package neureka.backend.api;

import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.FunPair;

public interface Fun {

    interface F64ToF64 extends Fun {
        double invoke(double x);

        static FunPair<F64ToF64> pair(F64ToF64 activtion, F64ToF64 deriviation) {
            return new Activation.Funs<>(activtion, deriviation);
        }
    }

    interface F32ToF32 extends Fun {
        float invoke(float x);

        static FunPair<F32ToF32> pair(F32ToF32 activtion, F32ToF32 deriviation) {
            return new Activation.Funs<>(activtion, deriviation);
        }
    }

    //...


    interface F64F64ToF64 extends Fun {
        double invoke(double x, double y);
    }

    interface F32F32ToF32 extends Fun {
        float invoke(float x, float y);
    }


}
