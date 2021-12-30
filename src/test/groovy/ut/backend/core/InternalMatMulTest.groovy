package ut.backend.core

import groovy.transform.CompileStatic
import neureka.backend.standard.operations.linear.fast.Conf
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
        _basicF32Test(Type.ROW_MAJOR)
        _basicF32Test(Type.COL_MAJOR)

        Type.COL_MAJOR.set()

        _testDoubles()
        _testFloats()

    }

    private static void _testDoubles() {

        /*
         * Create 2 random matrices, large enough
         */
        MatrixF64 A = MatrixF64.FACTORY.make(DIM, DIM,  new double[DIM*DIM])
        MatrixF64 B = MatrixF64.FACTORY.make(DIM, DIM,  new double[DIM*DIM])

        _fillIt64(A.data, 43)
        _fillIt64(B.data, 87)

        /*
         * A runnable task that executes the multiplication
         */

        // Warmup:
        MatrixF64 c1 = _matmulF64(A,B)
        MatrixF64 a1 = A.addMat(c1)
        MatrixF64 m1 = A.subtract(c1)

        MatrixF64 s = A.subtract((Double)4d)
        assert _hash(s.getData()) == -2106189063

        A.divide(0.6d)

        assert _hash(c1.getData()) ==  245517809
        assert _hash(a1.getData()) == -962793700
        assert _hash(m1.getData()) ==  188015414

        assert Arrays.toString(m1.divide(0.3d).getData()).hashCode() == -2049253096

        System.out.println("DOUBLES DONE!")
    }

    private static void _testFloats() {

        /*
         * Create 2 random matrices, large enough
         */
        MatrixF32 A = MatrixF32.FACTORY.make(DIM, DIM, new float[DIM*DIM])
        MatrixF32 B = MatrixF32.FACTORY.make(DIM, DIM, new float[DIM*DIM])

        _fillIt32(A.data, 12)
        _fillIt32(B.data, 98)

        /*
         * A runnable task that executes the multiplication
         */

        MatrixF32 s = A.subtract((Double)4d)
        assert _hash(s.getData()) == 1671633773

        MatrixF32 c2 = _matmulF32(A,B)
        MatrixF32 a2 = A.addMat(c2)
        MatrixF32 m2 = A.subtract(c2)

        A.divide(0.6d)

        assert _hash(c2.getData()) == 1204091031
        assert _hash(a2.getData()) == -1389630508
        assert _hash(m2.getData()) == -1238245633

        assert _hash(m2.divide(0.3d).getData()) == 587244582

        System.out.println("FLOATS DONE!")
    }

    private static void _basicF64Test(Type type) {

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

        var D = C.add(7d)

        assert D.data == (type == Type.COL_MAJOR ? [-3, -3] : [2, 2] ) as double[]
    }


    private static void _basicF32Test(Type type) {

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

        var D = C.add(7d)

        assert D.data == (type == Type.COL_MAJOR ? [-3, -3] : [2, 2] ) as float[]
    }


    private static MatrixF32 _matmulF32(MatrixF32 A, MatrixF32 B) {
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
