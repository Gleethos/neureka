package neureka.backend.standard.algorithms.internal;

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

    interface I32ToI32 extends Fun {
        int invoke( int x );

        static FunArray<I32ToI32> pair( I32ToI32 activation, I32ToI32 derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface ObjToObj extends Fun {
        Object invoke( Object x );

        static FunArray<ObjToObj> pair( ObjToObj activation, ObjToObj derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface ObjObjToObj extends Fun {
        Object invoke( Object x, Object y );

        static FunArray<ObjObjToObj> triple( ObjObjToObj activation, ObjObjToObj derivation, ObjObjToObj derivation2 ) {
            return new Operator.FunTriple<>( activation, derivation, derivation2 );
        }

        static FunArray<ObjObjToObj> of( ObjObjToObj activation ) {
            return new Operator.FunTriple<>( activation, null, null );
        }
    }
    //...


    interface F64F64ToF64 extends Fun {
        double invoke( double x, double y );

        static FunArray<F64F64ToF64> triple( F64F64ToF64 activation, F64F64ToF64 derivation1, F64F64ToF64 derivation2 ) {
            return new Operator.FunTriple<>(activation, derivation1, derivation2);
        }

        static FunArray<F64F64ToF64> of( F64F64ToF64 activation ) {
            return new Operator.FunTriple<>(activation, null, null);
        }
    }

    interface F32F32ToF32 extends Fun {
        float invoke(float x, float y);

        static FunArray<F32F32ToF32> triple( F32F32ToF32 activation, F32F32ToF32 derivation1, F32F32ToF32 derivation2 ) {
            return new Operator.FunTriple<>(activation, derivation1, derivation2);
        }

        static FunArray<F32F32ToF32> of( F32F32ToF32 activation ) {
            return new Operator.FunTriple<>(activation, null, null);
        }
    }

    interface I32I32ToI32 extends Fun {
        int invoke( int x, int y );

        static FunArray<I32I32ToI32> triple(I32I32ToI32 activation, I32I32ToI32 derivation1, I32I32ToI32 derivation2 ) {
            return new Operator.FunTriple<>( activation, derivation1, derivation2 );
        }
    }

}
