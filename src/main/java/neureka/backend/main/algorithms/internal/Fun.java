package neureka.backend.main.algorithms.internal;

import neureka.backend.main.algorithms.ElementWise;

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

    interface I64ToI64 extends Fun
    {
        long invoke( long x );

        static FunTuple<I64ToI64> pair(I64ToI64 activation, I64ToI64 derivation ) {
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

    interface BoolToBool extends Fun
    {
        boolean invoke( boolean x );

        static FunTuple<BoolToBool> pair(BoolToBool activation, BoolToBool derivation ) {
            return new FunPair<>( activation, derivation );
        }
    }

    interface CharToChar extends Fun
    {
        char invoke( char x );

        static FunTuple<CharToChar> pair(CharToChar activation, CharToChar derivation ) {
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
            return new ElementWise.FunTriple<>( activation, derivation, derivation2 );
        }

        static FunTuple<ObjObjToObj> of(ObjObjToObj activation ) {
            return new ElementWise.FunTriple<>( activation, null, null );
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
            return new ElementWise.FunTriple<>(activation, derivation1, derivation2);
        }

        static FunTuple<F64F64ToF64> of(F64F64ToF64 activation ) {
            return new ElementWise.FunTriple<>(activation, null, null);
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
            return new ElementWise.FunTriple<>(activation, derivation1, derivation2);
        }

        static FunTuple<F32F32ToF32> of(F32F32ToF32 activation ) {
            return new ElementWise.FunTriple<>(activation, null, null);
        }
    }

    interface I32I32ToI32 extends Fun
    {
        int invoke( int x, int y );

        static FunTuple<I32I32ToI32> of(I32I32ToI32 activation ) {
            return new ElementWise.FunTriple<>(activation, null, null);
        }

        static FunTuple<I32I32ToI32> triple(
                I32I32ToI32 activation,
                I32I32ToI32 derivation1,
                I32I32ToI32 derivation2
        ) {
            return new ElementWise.FunTriple<>( activation, derivation1, derivation2 );
        }
    }

    interface I64I64ToI64 extends Fun
    {
        long invoke( long x, long y );

        static FunTuple<I64I64ToI64> of(I64I64ToI64 activation ) {
            return new ElementWise.FunTriple<>(activation, null, null);
        }

        static FunTuple<I64I64ToI64> triple(
                I64I64ToI64 activation,
                I64I64ToI64 derivation1,
                I64I64ToI64 derivation2
        ) {
            return new ElementWise.FunTriple<>( activation, derivation1, derivation2 );
        }
    }

    interface I8I8ToI8 extends Fun
    {
        byte invoke( byte x, byte y );

        static FunTuple<I8I8ToI8> of(I8I8ToI8 activation ) {
            return new ElementWise.FunTriple<>(activation, null, null);
        }

        static FunTuple<I8I8ToI8> triple(
                I8I8ToI8 activation,
                I8I8ToI8 derivation1,
                I8I8ToI8 derivation2
        ) {
            return new ElementWise.FunTriple<>( activation, derivation1, derivation2 );
        }
    }

    interface BoolBoolToBool extends Fun
    {
        boolean invoke( boolean x, boolean y );

        static FunTuple<BoolBoolToBool> of(BoolBoolToBool activation ) {
            return new ElementWise.FunTriple<>(activation, null, null);
        }

        static FunTuple<BoolBoolToBool> triple(
                BoolBoolToBool activation,
                BoolBoolToBool derivation1,
                BoolBoolToBool derivation2
        ) {
            return new ElementWise.FunTriple<>( activation, derivation1, derivation2 );
        }
    }

    interface CharCharToChar extends Fun
    {
        char invoke( char x, char y );

        static FunTuple<CharCharToChar> of(CharCharToChar activation ) {
            return new ElementWise.FunTriple<>(activation, null, null);
        }

        static FunTuple<CharCharToChar> triple(
                CharCharToChar activation,
                CharCharToChar derivation1,
                CharCharToChar derivation2
        ) {
            return new ElementWise.FunTriple<>( activation, derivation1, derivation2 );
        }
    }
}
