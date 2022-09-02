package ut.backend.core

import groovy.transform.CompileStatic
import neureka.backend.main.operations.linear.internal.M32
import neureka.backend.main.operations.linear.internal.M64

@CompileStatic
class InternalMatMulTest
{
    private boolean DO_ROW_MAJOR = true

    private enum Type {
        ROW_MAJOR, COL_MAJOR;

        boolean set() { return this == ROW_MAJOR  }
    }

    private static final int DIM = 1_028

    static void main(final String[] args) {

        new InternalMatMulTest().start()

    }

    void start() {

        DO_ROW_MAJOR = Type.COL_MAJOR.set()
        _test(11,7,13,-1572171092)

        DO_ROW_MAJOR = Type.COL_MAJOR.set()
        _test(2,2,2,1621312579)
        _test(3,3,3,458449078)
        _test(4,4,4,-1394046135)
        _test(5,5,5,726435157)
        _test(6,6,6,952800666)
        _test(7,7,7,478609247)
        _test(8,8,8,-1274930005)
        _test(9,9,9,1754156081)
        _test(10,10,10,1703295132)

        _test(1,1,10,-1194484459)
        _test(1,1,9,-242625105)
        _test(1,1,8,-1060007964)
        _test(1,1,7,1047354419)
        _test(1,1,6,606279625)
        _test(1,1,5,800495719)
        _test(1,1,4,-605917383)
        _test(1,1,3,1703409186)
        _test(1,1,2,-1782294977)

        _test(1,1,1,-1641913795)

        _test(2,1,1,-1782294977)
        _test(3,1,1,1703409186)
        _test(4,1,1,-605917383)
        _test(5,1,1,800495719)
        _test(6,1,1,606279625)
        _test(7,1,1,1047354419)
        _test(8,1,1,-1060007964)
        _test(9,1,1,-242625105)
        _test(10,1,1,-1194484459)
        _test(1,2,1,-1639292187)
        _test(1,3,1,-1638189920)
        _test(1,4,1,-1637296190)
        _test(1,5,1,-1637177026)
        _test(1,6,1,-1636521624)
        _test(1,7,1,-1636521624)
        _test(1,8,1,-1636491833)
        _test(1,9,1,-1636372669)
        _test(1,10,1,-1635478939)
        _test(DIM,DIM,DIM,245517809)

        //---
        DO_ROW_MAJOR = Type.ROW_MAJOR.set()
        _test(2,3,4, -34070530)
        _test(20,30,40, -237450937)
        _test(42,61,53, 1490187176)

        //---
        _basicF64Test(Type.ROW_MAJOR)
        _basicF64Test(Type.COL_MAJOR)
        _basicF32Test(Type.ROW_MAJOR)
        _basicF32Test(Type.COL_MAJOR)
    }

    private void _test(int d1, int d2, int d3, int hash1) {
        _testDoubles(d1,d2,d3,hash1)
        _testFloats(d1,d2,d3,hash1)
    }

    private void _testDoubles(int dim, int dim2, int dim3, int hash) {

        double[] aData = new double[dim*dim2]
        double[] bData = new double[dim2*dim3]

        _fillIt64(aData, 43)
        _fillIt64(bData, 87)

        var A = new M64(dim, dim2,  aData)
        var B = new M64(dim2, dim3,  bData)

        var C = _matmulF64(A,B)

        assert _hash(C.getData()) ==  hash
    }

    private void _testFloats(int dim, int dim2, int dim3, int hash) {

        float[] aData = new float[dim*dim2]
        float[] bData = new float[dim2*dim3]

        _fillIt32(aData, 43)
        _fillIt32(bData, 87)

        var A = new M32(dim, dim2, aData)
        var B = new M32(dim2, dim3, bData)

        var C = _matmulF32(A,B)

        assert _hash(C.getData()) == hash
    }

    private void _basicF64Test(Type type) {

        DO_ROW_MAJOR = type.set()

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

    private void _basicF32Test(Type type) {

        DO_ROW_MAJOR = type.set()

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

    private M32 _matmulF32(M32 A, M32 B) {
        var data = new float[A.getRowDim()*B.getColDim()]
        var C = A.multiply(DO_ROW_MAJOR, B, data)
        assert C.data === data
        return C
    }

    private M64 _matmulF64(M64 A, M64 B) {
        var data = new double[A.getRowDim()*B.getColDim()]
        var C = A.multiply(DO_ROW_MAJOR, B, data)
        assert C.data === data
        return C
    }

    private static int _hash(Object data) {
        if ( data instanceof float[] )
            return Arrays.toString((float[])data).hashCode()
        else if ( data instanceof double[] )
            return Arrays.toString((double[])data).hashCode()
        else
            return data.hashCode()
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
