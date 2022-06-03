package neureka.backend.main.algorithms.internal;

import neureka.backend.main.algorithms.Operator;

public interface Fun
{

    interface F64ToF64 extends Fun
    {
        double invoke( double x );

        static FunPair<F64ToF64> pair( F64ToF64 activation, F64ToF64 derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface F32ToF32 extends Fun
    {
        float invoke( float x );

        static FunTuple<F32ToF32> pair(F32ToF32 activation, F32ToF32 derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface I32ToI32 extends Fun
    {
        int invoke( int x );

        static FunTuple<I32ToI32> pair(I32ToI32 activation, I32ToI32 derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface I8ToI8 extends Fun
    {
        byte invoke( byte x );

        static FunTuple<I8ToI8> pair(I8ToI8 activation, I8ToI8 derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface I16ToI16 extends Fun
    {
        short invoke( short x );

        static FunTuple<I16ToI16> pair(I16ToI16 activation, I16ToI16 derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface ObjToObj extends Fun
    {
        Object invoke( Object x );

        static FunTuple<ObjToObj> pair(ObjToObj activation, ObjToObj derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface ObjObjToObj extends Fun
    {
        Object invoke( Object x, Object y );

        static FunTuple<ObjObjToObj> triple(
                ObjObjToObj activation,
                ObjObjToObj derivation,
                ObjObjToObj derivation2
        ) {
            return new Operator.FunTriple<>( activation, derivation, derivation2 );
        }

        static FunTuple<ObjObjToObj> of(ObjObjToObj activation ) {
            return new Operator.FunTriple<>( activation, null, null );
        }
    }

    interface F64F64ToF64 extends Fun
    {
        double invoke( double x, double y );

        static FunTuple<F64F64ToF64> triple(
                F64F64ToF64 activation,
                F64F64ToF64 derivation1,
                F64F64ToF64 derivation2
        ) {
            return new Operator.FunTriple<>(activation, derivation1, derivation2);
        }

        static FunTuple<F64F64ToF64> of(F64F64ToF64 activation ) {
            return new Operator.FunTriple<>(activation, null, null);
        }
    }

    interface F32F32ToF32 extends Fun
    {
        float invoke(float x, float y);

        static FunTuple<F32F32ToF32> triple(
                F32F32ToF32 activation,
                F32F32ToF32 derivation1,
                F32F32ToF32 derivation2
        ) {
            return new Operator.FunTriple<>(activation, derivation1, derivation2);
        }

        static FunTuple<F32F32ToF32> of(F32F32ToF32 activation ) {
            return new Operator.FunTriple<>(activation, null, null);
        }
    }

    interface I32I32ToI32 extends Fun
    {
        int invoke( int x, int y );

        static FunTuple<I32I32ToI32> triple(
                I32I32ToI32 activation,
                I32I32ToI32 derivation1,
                I32I32ToI32 derivation2
        ) {
            return new Operator.FunTriple<>( activation, derivation1, derivation2 );
        }
    }

}