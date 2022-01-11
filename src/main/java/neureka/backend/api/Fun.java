package neureka.backend.api;

import neureka.backend.standard.algorithms.FunArray;
import neureka.backend.standard.algorithms.FunPair;
import neureka.backend.standard.algorithms.Operator;

public interface Fun {

    interface F64ToF64 extends Fun {
        double invoke( double x );

        static FunPair<F64ToF64> pair( F64ToF64 activation, F64ToF64 derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface F32ToF32 extends Fun {
        float invoke( float x );

        static FunArray<F32ToF32> pair( F32ToF32 activation, F32ToF32 derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    //...


    interface F64F64ToF64 extends Fun {
        double invoke( double x, double y );

        static FunArray<F64F64ToF64> triple( F64F64ToF64 activation, F64F64ToF64 derivation1, F64F64ToF64 derivation2 ) {
            return new Operator.FunTriple<>(activation, derivation1, derivation2);
        }
    }

    interface F32F32ToF32 extends Fun {
        float invoke(float x, float y);

        static FunArray<F32F32ToF32> triple( F32F32ToF32 activation, F32F32ToF32 derivation1, F32F32ToF32 derivation2 ) {
            return new Operator.FunTriple<>(activation, derivation1, derivation2);
        }
    }

}
