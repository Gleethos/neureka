import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.acceleration.opencl.OpenCLDevice
import neureka.acceleration.opencl.OpenCLPlatform
import neureka.acceleration.opencl.utility.DeviceQuery
import neureka.gui.swing.AbstractSpaceMap
import neureka.gui.swing.GraphBoard
import neureka.gui.swing.SurfaceObject
import org.junit.Test
import util.DummyDevice
import util.NTester
import util.NTester_Tensor

class GroovyTests
{

    @Test
    void testVisualizer()
    {
        //=========================================================================
        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
        //=========================================================================

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s =  (a*b) + 2
        Tsr x = new Tsr([s * (s+c)], "th(I[0])")

        GraphBoard w = new GraphBoard(x)
        def map = w.getBuilder().getSurface().getMap()
        Thread.sleep(3000)
        def things = map.getAll()
        assert things.size()>1
        double[] frame = new double[4]
        frame[0] = -4000000
        frame[1] = +4000000
        frame[2] = -4000000
        frame[3] = +4000000
        //assert map.getAllWithin(frame).size()>1
        map = map.removeAndUpdate(map.getAllWithin(frame).get(0))
        assert map!=null
        assert map.getAll().size()<things.size()
        def action = new AbstractSpaceMap.MapAction(){
            @Override
            boolean act(SurfaceObject o){
                return true;
            }
        }
        assert map.findAllWithin(frame, action).size()>1
        Thread.sleep(5000)
        //while(true){
        //
        //}
    }


    @Test
    void testNetworkLegacyIndexing() {
        Neureka.Settings.Indexing.setLegacy(true)
        Tsr i_a = new Tsr([2, 1], [1, 2])
        Tsr w_a = new Tsr([2, 2], [1, 3, 4, -1]).setRqsGradient(true)
        Tsr o_a = new Tsr(i_a, "x", w_a)
        //[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0),
        //---
        Tsr w_b = new Tsr([2, 2], [-2, 1, 2, -1]).setRqsGradient(true)
        Tsr o_b = new Tsr(o_a, "x", w_b)
        //[2x1]:(-10.0, 5.0); ->d[2x2]:(-2.0, 1.0, 2.0, -1.0):g:(null), ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), ,
        //---
        Tsr w_c = new Tsr([2, 2], [0.5, 3, -2, -0.5]).setRqsGradient(true)
        Tsr o_c = new Tsr(o_a, "x", w_c)
        //[2x1]:(-0.5, 20.0); ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), , ->d[2x2]:(0.5, 3.0, -2.0, -0.5):g:(null),
        //---
        Tsr out = o_b * o_c

        assert o_a.toString().contains("(7.0, 2.0)")
        assert out.toString().contains("(5.0, 100.0)")
        assert o_b.toString().contains("(-10.0, 5.0)")
        assert o_c.toString().contains("(-0.5, 20.0)")

        assert w_a.toString().contains("g:(null)")
        assert w_b.toString().contains("g:(null)")

        out.backward(new Tsr([2, 1], 1))

        assert w_a.toString().contains("g:(null)")
        assert !w_b.toString().contains("g:(null)")
        Neureka.Settings.AD._applyGradientUntilTensorIsUsed = true
        w_a * 3
        Neureka.Settings.AD._applyGradientUntilTensorIsUsed = false
        assert w_a.toString().contains("g:(null)")
        assert !w_a.toString().contains("1.0, 3.0, 4.0, -1.0")
        assert !w_b.toString().contains("g:(null)")
        //TODO: calculate derivatives and errors and check correctness!
    }

    @Test
    void testNetwork()
    {
        Neureka.Settings.Indexing.setLegacy(false)

        Tsr i_a = new Tsr([2, 1], [1, 2])
        Tsr w_a = new Tsr([2, 2], [1, 3, 4, -1]).setRqsGradient(true)
        Tsr o_a = new Tsr(i_a,"x", w_a)
        //[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0),
        //---
        Tsr w_b = new Tsr([2, 2], [-2, 1, 2, -1]).setRqsGradient(true)
        Tsr o_b = new Tsr(o_a,"x", w_b)
        //[2x1]:(-10.0, 5.0); ->d[2x2]:(-2.0, 1.0, 2.0, -1.0):g:(null), ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), ,
        //---
        Tsr w_c = new Tsr([2, 2], [0.5, 3, -2, -0.5]).setRqsGradient(true)
        Tsr o_c = new Tsr(o_a, "x", w_c)
        //[2x1]:(-0.5, 20.0); ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), , ->d[2x2]:(0.5, 3.0, -2.0, -0.5):g:(null),
        //---
        Tsr out = o_b*o_c

        assert o_a.toString().contains("(9.0, 1.0)")
        assert out.toString().contains("(-127.5, -314.5)")
        assert o_b.toString().contains("(-17.0, 17.0)")
        assert o_c.toString().contains("(7.5, -18.5)")

        assert w_a.toString().contains("g:(null)")
        assert w_b.toString().contains("g:(null)")

        out.backward(new Tsr([2, 1], 1))

        assert w_a.toString().contains("g:(null)")
        assert !w_b.toString().contains("g:(null)")
        Neureka.Settings.AD._applyGradientUntilTensorIsUsed = true
        w_a * 3
        Neureka.Settings.AD._applyGradientUntilTensorIsUsed = false
        assert w_a.toString().contains("g:(null)")
        assert !w_a.toString().contains("1.0, 3.0, 4.0, -1.0")
        assert !w_b.toString().contains("g:(null)")
        //TODO: calculate derivatives and errors and check correctness!
    }

    @Test
    void testSlicing()
    {
        NTester_Tensor tester = new NTester_Tensor("Tensor tester (only cpu)")
        Device device = new DummyDevice()

        Neureka.Settings.Indexing.setLegacy(true)
        _testReadmeExamples(device, tester, true)

        Neureka.Settings.Indexing.setLegacy(false)
        _testReadmeExamples(device, tester, false)

        //=========================================================================
        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
        //=========================================================================
        Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0)

        Neureka.Settings.Indexing.setLegacy(true)
        OpenCLPlatform.PLATFORMS().get(0).recompile()
        _testReadmeExamples(gpu, tester, true)

        Neureka.Settings.Indexing.setLegacy(false)
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
        tester.testContains(a.toString(),
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
                ], "Testing slicing")
        System.out.println("Done")
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
        tester.testContains(a.toString(), [
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
        ], "Testing slicing")
        System.out.println("Done")
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
        System.out.println(b.toString())
        x = new Tsr(b, "x", c)//This test is important!
        tester.testContains(x.toString(),
                [
                        (legacyIndexing)?
                        "[1x1]:(33.0); ->d[2x2]:(-2.0, 3.0, 1.0, 2.0)"
                        :"[1x1]:(20.0); ->d[2x2]:(-2.0, 3.0, 1.0, 2.0)"

                ], "")

    }

    @Test
    void testNN(){

        Device device = new DummyDevice()
        _testNN(device)

        //=========================================================================
        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
        //=========================================================================

        Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0)
        _testNN(gpu)

    }

    void _testNN(Device device)
    {
        Neureka.Settings.Indexing.setLegacy(false)

        Tsr X = new Tsr(// input data: 5 vectors in binary form
                [5, 3, 1],
                [
                        0, 0, 1,
                        1, 1, 0,
                        1, 0, 1,
                        0, 1, 1,
                        1, 1, 1
                ]
        ).add(device)

        Tsr y = new Tsr(// output values (labels)
                [5, 1, 1],[0,1,1,1,0]
        ).add(device)// [1, 5, 1],(0,1,1,1,0)

        Tsr input = X
        Tsr weights1 = new Tsr([1, input.shape()[1], 4],
                [4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01,
                 1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01,
                 3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01]
        ).add(device)
        /*
            [1x5x4]:(...)
            w1 (3, 4) :
            [[4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01]
             [1.46755891e-01 9.23385948e-02 1.86260211e-01 3.45560727e-01]
             [3.96767474e-01 5.38816734e-01 4.19194514e-01 6.85219500e-01]]
        */
        Tsr weights2 = new Tsr([1, 1, 4], [0.20445225, 0.87811744, 0.02738759, 0.67046751]).add(device)
        /*
            [1x1x4]:...
            w2 (4, 1) :
            [[0.20445225]
             [0.87811744]
             [0.02738759]
             [0.67046751]]
         */
        Tsr output = new Tsr(y.shape(), [0.0, 0.0, 0.0, 0.0, 0.0]).add(device)
        /*
            out (5, 1) :
            [[0.0]
             [0.0]
             [0.0]
             [0.0]
             [0.0]]
        */
        Tsr layer1 = new Tsr()
        /*
              inp (5, 3) :
              [[0, 0, 1]
               [1, 1, 0]
               [1, 0, 1]
               [0, 1, 1]
               [1, 1, 1]]
         */

        // iterate 1500 times
        for( i in 0..1500){
            feedforward(weights1, weights2, input, output, layer1)
            backprop(weights1, weights2, input, output, layer1, y)
        }
        println('Input:'+ X)
        println('Expected output:'+ y)
        println('Predicted output:'+ output)

        assert output.value64()[0]>=0.0&&output.value64()[0]<=1.0
        assert output.value64()[1]>=0.0&&output.value64()[1]<=1.0
        assert output.value64()[2]>=0.0&&output.value64()[2]<=1.0

        assert output.value64()[0]>=0.0&&output.value64()[0]<=0.1
        assert output.value64()[1]>=0.95&&output.value64()[1]<=1.0
        assert output.value64()[2]>=0.95&&output.value64()[2]<=1.0
        assert output.value64()[3]>=0.95&&output.value64()[3]<=1.0
        assert output.value64()[4]>=0.0&&output.value64()[4]<=0.1

        assert output.value64()[0]>=0.0&&output.value64()[0]<=0.0022
        assert output.value64()[1]>=0.98&&output.value64()[1]<=1.0
        assert output.value64()[2]>=0.98&&output.value64()[2]<=1.0
        assert output.value64()[3]>=0.98&&output.value64()[3]<=1.0
        assert output.value64()[4]>=0.0&&output.value64()[4]<=0.026

    }

    Tsr sigmoid(Tsr x) {
        return new Tsr(x, "sig(I[0])")
        //return new Tsr(((Tsr.fcn.create.E(x.shape())**(-x))+1), "1/I[0]")//1.0 / (1 + Tsr.fcn.create.E(x.shape())**(-x))
    }

    Tsr sigmoid_derivative(Tsr x) {
        return x * (-x + 1)
    }

    void feedforward(Tsr weights1, Tsr weights2, Tsr input, Tsr output, Tsr layer1) {
        Tsr in0 = new Tsr([input, weights1], "i0xi1")
        layer1[] = sigmoid(in0)
        //println(layer1.toString("shp")+"=sig(  I"+input.toString("shp")+" X W"+weights1.toString("shp")+" )")
        Tsr in1 = new Tsr([layer1, weights2], "i0xi1")
        output[] = sigmoid(in1)
        //println(output.toString("shp")+"=sig( L1"+layer1.toString("shp")+" X W"+weights2.toString("shp")+" )\n")
    }

    void backprop(Tsr weights1, Tsr weights2, Tsr input, Tsr output, Tsr layer1, Tsr y) {
        // application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        Tsr delta = (y - output)*2
        Tsr derivative = delta*2*sigmoid_derivative(output)
        Tsr d_weights2 = new Tsr(
                [layer1, (derivative)],
                "i0xi1"
        )
        Tsr d_weights1 = new Tsr(
                [input, (new Tsr([derivative, weights2], "i0xi1") * sigmoid_derivative(layer1))],
                "i0xi1"
        )
        // update the weights with the derivative (slope) of the loss function
        weights1[] = weights1 + d_weights1
        weights2[] = weights2 + d_weights2
    }







}
