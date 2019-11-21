import neureka.core.Tsr
import neureka.core.device.Device
import neureka.core.device.openCL.OpenCLDevice
import neureka.core.device.openCL.OpenCLPlatform;
import neureka.core.function.Function;
import neureka.core.function.factory.assembly.FunctionBuilder;
import neureka.core.function.factory.autograd.GraphNode;
import org.junit.Test;
import util.NTester_Function;
import util.NTester_Tensor;


class TestExamples
{

    @Test
    void testOne()
    {
        Device device = new DummyDevice()
        _testReadmeExamples(device)

        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return

        Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0)
        _testReadmeExamples(gpu)
    }

    void _testReadmeExamples(Device device)
    {
        NTester_Tensor tester = new NTester_Tensor("Tensor tester (only cpu)")

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
        tester.testTensor(y, "[1]:(4.0); ->d[1]:(-8.0), ")
        y.backward(new Tsr(2))
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
        device.add(a)
        b = a[[1, -2]]
        tester.testContains(b.toString(), ["2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"], "Testing slicing")
        tester.testContains(((b.has(int[].class))?"Has index component":""), ["Has index component"], "Check if index component is present!")
        b = a[1, -2]
        tester.testContains(b.toString(), ["2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"], "Testing slicing")
        tester.testContains(((b.has(int[].class))?"Has index component":""), ["Has index component"], "Check if index component is present!")
        /**
         * 2, 3, 4,
         * 6, 7, 8,
         * 1, 2, 3,
         * 5, 6, 7,
         *
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
        tester.testContains(b.toString(), ["12.0, 3.0, 4.0, 6.0, 7.0, 16.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"], "Testing slicing")
        //tester.testTensor(x, ["-16.0"])
        //---
        Tsr c = new Tsr([3, 4], [
                -3, 2, 3,
                5, 6, 2,
                -1, 1, 2,
                3, 4, 2,
        ])
        //device.add(c)
        Tsr d = b + c
        tester.testContains(d.toString(),
                [
                        "9.0, 5.0, 7.0, " +
                         "11.0, 13.0, 18.0, " +
                         "0.0, 3.0, 5.0, " +
                         "8.0, 10.0, 9.0"
                ],
                "Testing slicing")
        //---
        b = a[1..3, 2..4]
        tester.testContains(b.toString(), ["1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 1.0, 2.0"], "Testing slicing")
        tester.testContains(((b.has(int[].class))?"Has index component":""), ["Has index component"], "Check if index component is present!")
        /**
         1, 12, 3, 4,
         5, 6, 7, 16,
         9, 1, 2, 3, => 1, 2, 3,
         4, 5, 6, 7, => 5, 6, 7,
         8, 9, 1, 2, => 9, 1, 2,
         3, 4, 5, 6
         */
        //---
        b = a[[[0..3]:2, [1..4]:2]]
        tester.testContains(b.toString(), ["5.0, 7.0, 4.0, 6.0"], "Testing slicing")
        tester.testContains(((b.has(int[].class))?"Has index component":""), ["Has index component"], "Check if index component is present!")
        /**
         1, 12, 3, 4,
         5, 6, 7, 16, => 5,  7,
         9, 1, 2, 3,
         4, 5, 6, 7, => 4,  6,
         8, 9, 1, 2,
         3, 4, 5, 6
         */
        //---
        Tsr p = new Tsr([2,2], [2, 55, 4, 7]).add((device instanceof DummyDevice)?null:device)
        Tsr u = new Tsr([2,2], [5, 2, 7, 34]).add((device instanceof DummyDevice)?null:device)

        p[] = u
        tester.testContains(p.toString(), ["5.0, 2.0, 7.0, 34.0"], "Testing slicing")

        //---
        a[[[0..3]:2, [1..4]:2]] = new Tsr([2, 2], [1, 2, 3, 4])
        tester.testContains(b.toString(), ["1.0, 2.0, 3.0, 4.0"], "Testing slicing")
        tester.testContains(a.toString(), ["1.0, 12.0, 3.0, 4.0, 1.0, 6.0, 2.0, 16.0, 9.0, 1.0, 2.0, 3.0, 3.0, 5.0, 4.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0"], "Testing slicing")
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
        tester.testContains(b.toString(), ["1.0, 8.0, 3.0, 4.0"], "Testing slicing")
        tester.testContains(a.toString(), ["1.0, 12.0, 3.0, 4.0, 1.0, 8.0, 8.0, 16.0, 9.0, 8.0, 8.0, 3.0, 3.0, 5.0, 4.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0"], "Testing slicing")
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
        //b>>
        //      1, 8,
        //      3, 4
        b.setRqsGradient(true)
        c = new Tsr([2, 2], [
                -2, 3,//-2 + 24 + 3 + 8
                1, 2,
        ])
        device.add(b).add(c)
        System.out.println(b.toString())
        x = new Tsr(b, "x", c)//This test is important!
        tester.testContains(x.toString(), ["[1x1]:(33.0); ->d[2x2]:(-2.0, 3.0, 1.0, 2.0)"], "")

    }

}
