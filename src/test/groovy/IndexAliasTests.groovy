import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.acceleration.opencl.OpenCLDevice
import neureka.acceleration.opencl.OpenCLPlatform
import neureka.acceleration.opencl.utility.DeviceQuery
import neureka.autograd.GraphNode
import org.junit.Test
import util.DummyDevice
import util.NTester
import util.NTester_Tensor

class IndexAliasTests
{

    @Test
    void testSlicing()
    {
        Neureka.instance().settings().reset()
        Neureka.instance().settings().view().setLegacy(true)

        NTester_Tensor tester = new NTester_Tensor("IndexAlias-Testing: slices/subset creation and calculation")
        Device device = new DummyDevice()

        Neureka.instance().settings().indexing().setLegacy(true)
        _testReadmeExamples(device, tester, true)

        Neureka.instance().settings().indexing().setLegacy(false)
        _testReadmeExamples(device, tester, false)

        //=========================================================================
        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
        //=========================================================================
        Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0)

        Neureka.instance().settings().indexing().setLegacy(true)
        OpenCLPlatform.PLATFORMS().get(0).recompile()
        _testReadmeExamples(gpu, tester, true)

        Neureka.instance().settings().indexing().setLegacy(false)
        OpenCLPlatform.PLATFORMS().get(0).recompile()
        _testReadmeExamples(gpu, tester, false)

        String query = DeviceQuery.query()
        assert query.contains("DEVICE_NAME")
        assert query.contains("MAX_MEM_ALLOC_SIZE")
        assert query.contains("VENDOR")
        assert query.contains("CL_DEVICE_PREFERRED_VECTOR_WIDTH")
        assert query.contains("Info for device")
        assert query.contains("LOCAL_MEM_SIZE")
        assert query.contains("CL_DEVICE_TYPE")

        OpenCLDevice cld = (OpenCLDevice)gpu;
        assert cld.globalMemSize()>1000
        assert !cld.name().equals("")
        assert cld.image2DMaxHeight()>100
        assert cld.image3DMaxHeight()>100
        assert cld.maxClockFrequenzy()>100
        assert !cld.vendor().equals("")
        assert !cld.toString().equals("")
        assert cld.maxConstantBufferSize()>1000
        assert cld.maxWriteImageArgs()>1

        tester.close()
    }

    void _testReadmeExamples(Device device, NTester tester, boolean legacyIndexing)
    {
        Tsr x = new Tsr([1], 3).setRqsGradient(true)
        Tsr b = new Tsr([1], -4)
        Tsr w = new Tsr([1], 2)
        device.add(x).add(b).add(w)
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        Tsr y = new Tsr([x, b, w], "((i0+i1)*i2)^2")
        tester.testContains((y.idxmap()==null)?"true":"false", ["false"], "idxmap must be set!")
        tester.testTensor(y, "[1]:(4.0); ->d[1]:(-8.0)")
        y.backward(new Tsr(2))

        y = ((x+b)*w)**2
        tester.testTensor(y, ["[1]:(4.0); ->d[1]:(-8.0)"])
        y.backward(new Tsr(2))
        tester.testTensor(x, ["-32.0"])

        y = b + w * x

        /**
         *  Subset:
         */
        Tsr a = new Tsr([4, 6], [
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 1, 2,
                3, 4, 5, 6
        ])
        /* //NON LEGACY INDEXED:
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 1, 2, 3, => 7, 8, 9, 1
            4, 5, 6, 7, 8, 9, => 4, 5, 6, 7
            1, 2, 3, 4, 5, 6  => 1, 2, 3, 4
         */

        device.add(a)
        b = a[[-1..-3, -6..-3]]
        def s = a[[1, -2]]
        assert s==((legacyIndexing)?9.0:2.0)
        tester.testContains(b.toString(),
                [
                        (legacyIndexing)
                        ?"2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"
                        :"7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
                ], "Testing slicing")
        tester.testContains(((b.has(int[].class))?"Has index component":""), ["Has index component"], "Check if index component is present!")
        b = a[-3..-1, 0..3]
        s = a[1, -2]
        assert s==((legacyIndexing)?9.0:2.0)
        tester.testContains(b.toString(),
                [
                        (legacyIndexing)
                        ?"2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"
                        :"7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
                ], "Testing slicing")
        tester.testContains(((b.has(int[].class))?"Has index component":""), ["Has index component"], "Check if index component is present!")
        /**
         * 2, 3, 4,
         * 6, 7, 8,
         * 1, 2, 3,
         * 5, 6, 7,
         */
        //---
        if(device instanceof DummyDevice){
            a.value64()[1] = a.value64()[1] * 6
            a.value64()[7] = a.value64()[7] * 2
        }else{
            Tsr k = new Tsr([4, 6], [
                    1, 6, 1, 1,
                    1, 1, 1, 2,
                    1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1
            ])
            device.add(k)
            a[] = a*k
        }
        tester.testContains(b.toString(), [
                (legacyIndexing)
                ?"12.0, 3.0, 4.0, 6.0, 7.0, 16.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"
                :"7.0, 16.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
        ], "Testing slicing")
        //tester.testTensor(x, ["-16.0"])
        //---
        Tsr c = new Tsr([3, 4], [
                -3, 2, 3,
                5, 6, 2,
                -1, 1, 2,
                3, 4, 2,
        ])
        /* //NON LEGACY INDEXED:
            -3, 2, 3, 5,
            6, 2, -1, 1,
            2, 3, 4, 2,
                +
            7, 18, 9, 1
            4, 5, 6, 7
            1, 2, 3, 4
                =
            4, 20, 12, 6
            10, 7, 5,  8
            3,  5, 7   6

         */
        //device.add(c)
        Tsr d = b + c
        tester.testContains(d.toString(),
                [
                        (legacyIndexing)?
                        "9.0, 5.0, 7.0, " +
                         "11.0, 13.0, 18.0, " +
                         "0.0, 3.0, 5.0, " +
                         "8.0, 10.0, 9.0"
                        :"4.0, 18.0, 12.0, 6.0, "+
                         "10.0, 7.0, 5.0, 8.0, "+
                         "3.0, 5.0, 7.0, 6.0"
                ],
                "Testing slicing"
        )
        //---
        b = a[1..3, 2..4]
        tester.testContains(b.toString(),
                [
                    (legacyIndexing)
                        ?"1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 1.0, 2.0"
                        :"9.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0"
                ],
                "Testing slicing"
        )
        tester.testContains(((b.has(int[].class))?"Has index component":""), ["Has index component"], "Check if index component is present!")
        /**
         1, 12, 3, 4,
         5, 6, 7, 16,
         9, 1, 2, 3, => 1, 2, 3,
         4, 5, 6, 7, => 5, 6, 7,
         8, 9, 1, 2, => 9, 1, 2,
         3, 4, 5, 6

         NON LEGACY INDEXING:
         1, 12, 3, 4, 5, 6,
         7, 16, 9, 1, 2, 3, =>  9, 1, 2,
         4, 5, 6, 7, 8, 9,  =>  6, 7, 8,
         1, 2, 3, 4, 5, 6   =>  3, 4, 5,
         */
        //---
        b = a[[[0..3]:2, [1..4]:2]]
        tester.testContains(b.toString(),
                [
                        (legacyIndexing)
                        ?"5.0, 7.0, 4.0, 6.0"
                        :"12.0, 4.0, 5.0, 7.0"
                ], "Testing slicing")
        tester.testContains(((b.has(int[].class))?"Has index component":""), ["Has index component"], "Check if index component is present!")
        /**
         1, 12, 3, 4,
         5, 6, 7, 16, => 5,  7,
         9, 1, 2, 3,
         4, 5, 6, 7, => 4,  6,
         8, 9, 1, 2,
         3, 4, 5, 6

         NON LEGACY INDEXING:
         1, 12, 3, 4, 5, 6, => 12, 4,
         7, 16, 9, 1, 2, 3,
         4, 5, 6, 7, 8, 9,  => 5,  7,
         1, 2, 3, 4, 5, 6

         */
        //---
        Tsr p = new Tsr([2,2], [2, 55, 4, 7]).add((device instanceof DummyDevice)?null:device)
        Tsr u = new Tsr([2,2], [5, 2, 7, 34]).add((device instanceof DummyDevice)?null:device)

        p[] = u
        tester.testContains(p.toString(), ["5.0, 2.0, 7.0, 34.0"], "Testing slicing")

        //---
        a[[[0..3]:2, [1..4]:2]] = new Tsr([2, 2], [1, 2, 3, 4])
        tester.testContains(b.toString(), ["1.0, 2.0, 3.0, 4.0"], "Testing slicing")
        tester.testContains(
                a.toString(),
                [
                        (legacyIndexing)
                        ?"1.0, 12.0, 3.0, 4.0, " +
                        "1.0, 6.0, 2.0, 16.0, " +
                        "9.0, 1.0, 2.0, 3.0, " +
                        "3.0, 5.0, 4.0, 7.0, " +
                        "8.0, 9.0, 1.0, 2.0, " +
                        "3.0, 4.0, 5.0, 6.0"
                        :"1.0, 1.0, 3.0, 2.0, 5.0, 6.0, " +
                         "7.0, 16.0, 9.0, 1.0, 2.0, 3.0, " +
                         "4.0, 3.0, 6.0, 4.0, 8.0, 9.0, " +
                         "1.0, 2.0, 3.0, 4.0, 5.0, 6.0"
                ],
                "Testing slicing"
        )
        /**a:>>
             1, 12, 3, 4,
             1, 6, 2, 16,
             9, 1, 2, 3,
             3, 5, 4, 7,
             8, 9, 1, 2,
             3, 4, 5, 6
         */
        //---

        a[1..2, 1..2] = new Tsr([2, 2], [8, 8, 8, 8])
        tester.testContains(b.toString(),
        [
                (legacyIndexing)?
                 "1.0, 8.0, " +
                 "3.0, 4.0"
                :"1.0, 2.0, "+
                 "8.0, 4.0"
        ], "Testing slicing")
        tester.testContains(
                a.toString(), [
                (legacyIndexing)?
                "1.0, 12.0, 3.0, 4.0, " +
                "1.0, 8.0, 8.0, 16.0, " +
                "9.0, 8.0, 8.0, 3.0, " +
                "3.0, 5.0, 4.0, 7.0, " +
                "8.0, 9.0, 1.0, 2.0, " +
                "3.0, 4.0, 5.0, 6.0"
                :"1.0, 1.0, 3.0, 2.0, 5.0, 6.0, " +
                 "7.0, 8.0, 8.0, 1.0, 2.0, 3.0, " +
                 "4.0, 8.0, 8.0, 4.0, 8.0, 9.0, " +
                 "1.0, 2.0, 3.0, 4.0, 5.0, 6.0"
        ],
                "Testing slicing"
        )
        /**a:>>
         1, 12, 3, 4,
         1, 8, 8, 16,
         9, 8, 8, 3,
         3, 5, 4, 7,
         8, 9, 1, 2,
         3, 4, 5, 6

         NON LEGACY INDEXING
         1.0, 1.0, 3.0, 2.0, 5.0, 6.0,
         7.0, 8.0, 8.0, 1.0, 2.0, 3.0,
         4.0, 8.0, 8.0, 4.0, 8.0, 9.0,
         1.0, 2.0, 3.0, 4.0, 5.0, 6.0
         */

        //---
        //b>>
        //      1, 8,
        //      3, 4
        b.setRqsGradient(true)
        c = new Tsr([2, 2], [
                -2, 3,//-2 + 24 + 3 + 8
                1, 2,
        ])
        device.add(b).add(c)//-2+6+8+8 = 22
        x = new Tsr(b, "x", c)//This test is important!
        tester.testContains(x.toString(),
                [
                        (legacyIndexing)?
                        "[1x1]:(33.0); ->d[2x2]:(-2.0, 3.0, 1.0, 2.0)"
                        :"[1x1]:(20.0); ->d[2x2]:(-2.0, 3.0, 1.0, 2.0)"

                ], "")

    }






}
