package ut.backend.core

import groovy.transform.CompileStatic
import neureka.backend.standard.operations.linear.fast.Conf
import neureka.backend.standard.operations.linear.fast.M32
import neureka.backend.standard.operations.linear.fast.M64

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

        Type.COL_MAJOR.set()
        _test(2,2,2,1621312579,-1622243452)
        _test(3,3,3,458449078,175514624)
        _test(4,4,4,-1394046135,1168903786)
        _test(5,5,5,726435157,1348725681)

        _test(1,1,10,-1194484459,155274511)
        _test(1,1,9,-242625105,72695794)
        _test(1,1,8,-1060007964,-1265014008)
        _test(1,1,7,1047354419,-1000173898)
        _test(1,1,6,606279625,-138392009)
        _test(1,1,5,800495719,1248567540)
        _test(1,1,4,-605917383,93003368)
        _test(1,1,3,1703409186,-1524639888)
        _test(1,1,2,-1782294977,853895268)

        _test(1,1,1,-1641913795,-1642807525)

        _test(2,1,1,-1782294977,853895268)
        _test(3,1,1,1703409186,-1524639888)
        _test(4,1,1,-605917383,93003368)
        _test(5,1,1,800495719,1248567540)
        _test(6,1,1,606279625,-138392009)
        _test(7,1,1,1047354419,-1000173898)
        _test(8,1,1,-1060007964,-1265014008)
        _test(9,1,1,-242625105,72695794)
        _test(10,1,1,-1194484459,155274511)
        _test(1,2,1,-1639292187,-1641913795)
        _test(1,3,1,-1638189920,-1641794631)
        _test(1,4,1,-1637296190,-1641139229)
        _test(1,5,1,-1637177026,-1641139229)
        _test(1,6,1,-1636521624,-1641109438)
        _test(1,7,1,-1636521624,-1640990274)
        _test(1,8,1,-1636491833,-1640096544)
        _test(1,9,1,-1636372669,-1638368666)
        _test(1,10,1,-1635478939,-1636372669)
        _test(DIM,DIM,DIM,245517809,1204091031)


        _basicF64Test(Type.ROW_MAJOR)
        _basicF64Test(Type.COL_MAJOR)
        _basicF32Test(Type.ROW_MAJOR)
        _basicF32Test(Type.COL_MAJOR)


    }

    private static void _test(int d1, int d2, int d3, int hash1, int hash2) {
        _testDoubles(d1,d2,d3,hash1)
        _testFloats(d1,d2,d3,hash2)
    }

    private static void _testDoubles(int dim, int dim2, int dim3, int hash) {

        var A = new M64(dim, dim2,  new double[dim*dim2])
        var B = new M64(dim2, dim3,  new double[dim2*dim3])

        _fillIt64(A.data, 43)
        _fillIt64(B.data, 87)

        var C = _matmulF64(A,B)

        assert _hash(C.getData()) ==  hash
    }

    private static void _testFloats(int dim, int dim2, int dim3, int hash) {

        var A = new M32(dim, dim2, new float[dim*dim2])
        var B = new M32(dim2, dim3, new float[dim2*dim3])

        _fillIt32(A.data, 12)
        _fillIt32(B.data, 98)

        var C = _matmulF32(A,B)

        assert _hash(C.getData()) == hash
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

    private static M32 _matmulF32(M32 A, M32 B) {
        var data = new float[A.getRowDim()*B.getColDim()]
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
