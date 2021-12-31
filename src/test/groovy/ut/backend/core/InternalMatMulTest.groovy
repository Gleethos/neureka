package ut.backend.core

import groovy.transform.CompileStatic
import neureka.backend.standard.operations.linear.fast.Conf
import neureka.backend.standard.operations.linear.fast.M32
import neureka.backend.standard.operations.linear.fast.M64
import neureka.backend.standard.operations.linear.fast.matrix.MatrixF32
import neureka.backend.standard.operations.linear.fast.matrix.MatrixF64

@CompileStatic
class InternalMatMulTest {

    private enum Type {
        ROW_MAJOR, COL_MAJOR;

        void set() {
            if ( this == ROW_MAJOR )
                Conf.ROW_MAJOR = true
            else
                Conf.ROW_MAJOR = false
        }
    }

    private static final int DIM = 1_028

    static void main(final String[] args) {

        start()

    }

    static void start() {

        _basicF64Test(Type.ROW_MAJOR)
        _basicF64Test(Type.COL_MAJOR)
        _basicF64TestOLD(Type.ROW_MAJOR)
        _basicF64TestOLD(Type.COL_MAJOR)
        _basicF32Test(Type.ROW_MAJOR)
        _basicF32Test(Type.COL_MAJOR)
        _basicF32TestOLD(Type.ROW_MAJOR)
        _basicF32TestOLD(Type.COL_MAJOR)

        Type.COL_MAJOR.set()

        _testDoubles()
        _testFloats()

    }

    private static void _testDoubles() {

        MatrixF64 A = MatrixF64.FACTORY.make(DIM, DIM,  new double[DIM*DIM])
        MatrixF64 B = MatrixF64.FACTORY.make(DIM, DIM,  new double[DIM*DIM])

        _fillIt64(A.data, 43)
        _fillIt64(B.data, 87)

        MatrixF64 C = _matmulF64(A,B)

        assert _hash(C.getData()) ==  245517809
    }

    private static void _testFloats() {

        MatrixF32 A = MatrixF32.FACTORY.make(DIM, DIM, new float[DIM*DIM])
        MatrixF32 B = MatrixF32.FACTORY.make(DIM, DIM, new float[DIM*DIM])

        _fillIt32(A.data, 12)
        _fillIt32(B.data, 98)

        MatrixF32 C = _matmulF32(A,B)

        assert _hash(C.getData()) == 1204091031

    }

    private static void _basicF64TestOLD(Type type) {

        type.set()

        //---

        MatrixF64 A = MatrixF64.FACTORY.make(1, 2,  new double[2])
        MatrixF64 B = MatrixF64.FACTORY.make(2, 2,  new double[4])
        _fillIt64(A.data, -339)
        _fillIt64(B.data, 543)

        var C =  _matmulF64(A,B)

        /*
                        ( 5  -5 )
                        (-4  -3 )
                (-5 -4) (-9  37 )

                        ( 5  -4 )
                        (-5  -3 )
                (-5 -4) (-5  32 )
        */

        assert A.data == [-5, -4] as double[]
        assert B.data == [5, -5, -4, -3] as double[]
        assert C.data == (type == Type.COL_MAJOR ? [-5, 32] : [-9, 37] ) as double[]

        //---

        A = MatrixF64.FACTORY.make(2, 2,  new double[4])
        B = MatrixF64.FACTORY.make(2, 1,  new double[2])
        _fillIt64(A.data, -339)
        _fillIt64(B.data, 543)

        C =  _matmulF64(A,B)

        /*
                        ( 5)
                        (-5)
                (-5 -4) (-5)
                (-3 -2) (-5)

                        ( 5)
                        (-5)
                (-5 -3) (-10)
                (-4 -2) (-10)
        */

        assert A.data == [-5, -4, -3, -2] as double[]
        assert B.data == [5, -5] as double[]
        assert C.data == (type == Type.COL_MAJOR ? [-10, -10] : [-5, -5] ) as double[]
    }


    private static void _basicF64Test(Type type) {

        type.set()

        //---

        M64 A = new M64(1, 2,  new double[2])
        M64 B = new M64(2, 2,  new double[4])
        _fillIt64(A.data, -339)
        _fillIt64(B.data, 543)

        var C =  _matmulF64(A,B)

        /*
                        ( 5  -5 )
                        (-4  -3 )
                (-5 -4) (-9  37 )

                        ( 5  -4 )
                        (-5  -3 )
                (-5 -4) (-5  32 )
        */

        assert A.data == [-5, -4] as double[]
        assert B.data == [5, -5, -4, -3] as double[]
        assert C.data == (type == Type.COL_MAJOR ? [-5, 32] : [-9, 37] ) as double[]

        //---

        A = new M64(2, 2,  new double[4])
        B = new M64(2, 1,  new double[2])
        _fillIt64(A.data, -339)
        _fillIt64(B.data, 543)

        C =  _matmulF64(A,B)

        /*
                        ( 5)
                        (-5)
                (-5 -4) (-5)
                (-3 -2) (-5)

                        ( 5)
                        (-5)
                (-5 -3) (-10)
                (-4 -2) (-10)
        */

        assert A.data == [-5, -4, -3, -2] as double[]
        assert B.data == [5, -5] as double[]
        assert C.data == (type == Type.COL_MAJOR ? [-10, -10] : [-5, -5] ) as double[]

    }



    private static void _basicF32TestOLD(Type type) {

        type.set()

        MatrixF32 A = MatrixF32.FACTORY.make(2, 2,  new float[4])
        MatrixF32 B = MatrixF32.FACTORY.make(2, 1,  new float[2])
        _fillIt32(A.data, -339)
        _fillIt32(B.data, 543)

        var C =  _matmulF32(A,B)

        /*
                        ( 5)
                        (-5)
                (-5 -4) (-5)
                (-3 -2) (-5)

                        ( 5)
                        (-5)
                (-5 -3) (-10)
                (-4 -2) (-5)
        */

        assert A.data == [-5, -4, -3, -2] as float[]
        assert B.data == [5, -5] as float[]
        assert C.data == (type == Type.COL_MAJOR ? [-10, -10] : [-5, -5] ) as float[]

    }

    private static void _basicF32Test(Type type) {

        type.set()

        M32 A = new M32(2, 2,  new float[4])
        M32 B = new M32(2, 1,  new float[2])
        _fillIt32(A.data, -339)
        _fillIt32(B.data, 543)

        var C =  _matmulF32(A,B)

        /*
                        ( 5)
                        (-5)
                (-5 -4) (-5)
                (-3 -2) (-5)

                        ( 5)
                        (-5)
                (-5 -3) (-10)
                (-4 -2) (-5)
        */

        assert A.data == [-5, -4, -3, -2] as float[]
        assert B.data == [5, -5] as float[]
        assert C.data == (type == Type.COL_MAJOR ? [-10, -10] : [-5, -5] ) as float[]

    }



    private static MatrixF32 _matmulF32(MatrixF32 A, MatrixF32 B) {
        var data = new float[A.getRowDim()*B.getColDim()]
        var C = A.multiply(B, data)
        assert C.data === data
        return C
    }


    private static M32 _matmulF32(M32 A, M32 B) {
        var data = new float[A.getRowDim()*B.getColDim()]
        var C = A.multiply(B, data)
        assert C.data === data
        return C
    }

    private static MatrixF64 _matmulF64(MatrixF64 A, MatrixF64 B) {
        var data = new double[A.getRowDim()*B.getColDim()]
        var C = A.multiply(B, data)
        assert C.data === data
        return C
    }

    private static M64 _matmulF64(M64 A, M64 B) {
        var data = new double[A.getRowDim()*B.getColDim()]
        var C = A.multiply(B, data)
        assert C.data === data
        return C
    }

    private static int _hash(Object data) {
        if ( data instanceof float[] )
            return Arrays.toString((float[])data).hashCode()
        else
            return Arrays.toString((double[])data).hashCode()
    }

    private static void _fillIt64( double[] data, int seed ) {
        long hash = (long)((32**seed) % 11)
        data.length.times {
            data[it] = ( (it + hash) % 11 - 5 )
        }
    }

    private static void _fillIt32( float[] data, int seed ) {
        long hash = (long)((32**seed) % 11)
        data.length.times {
            data[it] = ( (it + hash) % 11 - 5 )
        }
    }

}
