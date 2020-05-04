
import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.host.HostCPU
import neureka.acceleration.Device
import neureka.acceleration.opencl.OpenCLPlatform
import neureka.autograd.JITProp
import neureka.calculus.Function
import neureka.calculus.factory.assembly.FunctionBuilder
import org.junit.Test
import util.DummyDevice
import util.Utility

import java.lang.ref.WeakReference


class ThoroughGroovyTests
{

    @Test
    void dot_operation_reshapes_and_produces_valid_x_operation(){

        Tsr a = new Tsr([1, 4, 4, 1], [4..12])
        Tsr b = new Tsr([1, 3, 5, 2, 1], [-5..3])
        Tsr c = a.dot(b)
        assert c.toString().contains("(1x4x2x5x2x1)")
    }

    @Test
    void test_benchmark_script_and_simple_tensor_constructor(){

        Neureka.instance().reset()

        Tsr t = new Tsr([1, 3, 6])
        assert !t.toString().contains("empty")
        assert t.toString().contains("(1x3x6)")
        t = new Tsr([1, 3.0, 6])
        assert !t.toString().contains("empty")
        assert t.toString().contains("(1x3x6):[0.0, 0.0, 0.0")
        t = new Tsr([1, 3.3, 6])
        assert !t.toString().contains("empty")
        assert t.toString().contains("(3):[1.0, 3.3, 6.0]")

        String benchmark = Utility.readResource("benchmark.groovy", this)
        def session = new GroovyShell().evaluate(benchmark)
        def hash = ""
        String expected = "56b2eb74955e49cd777469c7dad0536e"
        session([
                    "iterations":1,
                    "sample_size":20,
                    "difficulty":15,
                    "intensifier":0
                ],
                null,
                HostCPU.instance(),
                tsr->{
                    hash = (hash+tsr.toString()).md5()
                }
        )
        assert hash==expected

        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return

        hash = ""
        session([
                "iterations":1,
                "sample_size":20,
                "difficulty":15,
                "intensifier":0
        ],
                null,
                Device.find("nvidia"),
                tsr->{
                    hash = (hash+tsr.toString()).md5()
                }
        )
        assert hash==expected

        //String currentDate = new SimpleDateFormat("dd-MM-yyyy").format(new Date())
        //session([
        //            "iterations":1,
        //            "sample_size":20,
        //            "difficulty":15,
        //            "intensifier":50
        //        ],
        //        "neureka_bench_GPU_"+currentDate+".csv",
        //        Device.find("nvidia"),
        //        tsr->{}
        //)
        //session([
        //            "iterations":1,
        //            "sample_size":20,
        //            "difficulty":15,
        //            "intensifier":50
        //        ],
        //        "neureka_bench_CPU_"+currentDate+".csv",
        //        HostCPU.instance(),
        //        tsr->{}
        //)
        //session([
        //            "iterations":1,
        //            "sample_size":100,
        //            "difficulty":500,
        //            "intensifier":0
        //        ],
        //        "neureka_bench_GPU_100x_cd100_"+currentDate+".csv",
        //        Device.find("nvidia"),
        //        tsr->{}
        //)
        //session([
        //            "iterations":1,
        //            "sample_size":500,
        //            "difficulty":5,
        //            "intensifier":0
        //        ],
        //        "neureka_bench_CPU_500x_cd5_"+currentDate+".csv",
        //        HostCPU.instance(),
        //        tsr->{}
        //)
    }

    @Test
    void test_asFunction_method_inside_String()
    {
        Neureka.instance().reset()
        Tsr a = new Tsr([1,2], [3, 2])
        Tsr b = new Tsr([2,1], [-1, 4])
        Tsr c = "I[0]xI[1]".asFunction().activate(new Tsr[]{a, b})
        assert c.toString().contains("(2x2):[-3.0, -2.0, 12.0, 8.0]")
        c = "I[0]xI[1]"[a, b]
        assert c.toString().contains("(2x2):[-3.0, -2.0, 12.0, 8.0]")
        c = "i0 x i1"%[a, b]
        assert c.toString().contains("(2x2):[-3.0, -2.0, 12.0, 8.0]")
        c = "i0"%a
        assert c.toString().contains("(1x2):[3.0, 2.0]")
    }

    @Test
    void test_manual_convolution()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(false)
        Tsr a = new Tsr([100, 100], 3..19)
        Tsr x = a[1..-2,0..-1]
        Tsr y = a[0..-3,0..-1]
        Tsr z = a[2..-1,0..-1]
        Tsr rowconvol = x + y + z//(98, 100) (98, 100) (98, 100)
        Tsr k = rowconvol[0..-1,1..-2]
        Tsr v = rowconvol[0..-1,0..-3]
        Tsr j = rowconvol[0..-1,2..-1]
        Tsr u = a[1..-2,1..-2]
        Tsr colconvol = k + v + j - 9 * u//(98, 98)+(98, 98)+(98, 98)-9*(98, 98)
        print(colconvol)
        String asStr = x.toString()
        assert asStr.contains("(98x100)")
        assert asStr.contains("):[18.0, 19.0, 3.0, 4.0, 5.0")
        asStr = y.toString()
        assert asStr.contains("(98x100)")
        assert asStr.contains("):[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0")
        asStr = z.toString()
        assert asStr.contains("(98x100)")
        asStr.contains("):[16.0, 17.0, 18.0, 19.0, 3.0")
        asStr = rowconvol.toString()
        assert asStr.contains("(98x100)")
        assert asStr.contains("):[37.0, 40.0, 26.0, 29.0, 15.0, 18.0")
        asStr = k.toString()
        assert asStr.contains("(98x98)")
        asStr.contains("):[40.0, 26.0, 29.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0")
        asStr = v.toString()
        assert asStr.contains("(98x98)")
        assert asStr.contains("):[37.0, 40.0, 26.0, 29.0, 15.0, 18.0, 21.0")
        asStr = j.toString()
        assert asStr.contains("(98x98)")
        asStr.contains("):[26.0, 29.0, 15.0, 18.0, 21.0, 24.0")
        asStr = u.toString()
        assert asStr.contains("(98x98)")
        assert asStr.contains("):[19.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, ")
        asStr = colconvol.toString()
        assert asStr.contains("(98x98)")
        asStr.contains("(98x98):[-68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, " +
                "-34.0, -68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, -34.0, " +
                "-68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, ... + 9554 more]")
    }

    @Test
    void test_auto_reshape_and_broadcast()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)
        Tsr a = new Tsr([2,2], 1..5)
        Tsr b = new Tsr([2,1], 3..4)
        Tsr c = new Tsr([2], 8..9).setRqsGradient(true)
        assert (a+b).toString().contains("(4.0, 5.0, 7.0, 8.0)")
        Tsr t = (a+c)
        assert t.toString().contains("(9.0, 11.0, 11.0, 13.0)")
        //t.backward(new Tsr([2, 2], [5, -2, 7, 3]))
        assert c.toString().contains("")
        Neureka.instance().settings().view().setLegacy(false)
        assert t.toString().contains("):[9.0, 11.0, 11.0, 13.0]")
    }

    @Test
    void test_tensor_constructors()
    {
        Tsr a = new Tsr(3)
        Tsr b = new Tsr(2)
        Tsr c = new Tsr(-1)
        Tsr t = new Tsr("1+", a, "*", b)
        assert t.toString().contains("7.0")
        t = new Tsr("1", "+", a, "*", b)
        assert t.toString().contains("7.0")
        t = new Tsr("(","1+", a,")", "*", b)
        assert t.toString().contains("8.0")
        t = new Tsr("(","1", "+", a,")", "*", b)
        assert t.toString().contains("8.0")
        t = new Tsr("(", c, "*3)+", "(","1+", a,")", "*", b)
        assert t.toString().contains("5.0")
        t = new Tsr("(", c, "*","3)+", "(","1+", a,")", "*", b)
        assert t.toString().contains("5.0")
        t = new Tsr("(", c, "*","3", ")+", "(","1+", a,")", "*", b)
        assert t.toString().contains("5.0")

        t = new Tsr([2, 2], [2, 4, 4])
        assert t.toString().contains("(2x2):[2.0, 4.0, 4.0, 2.0]")
        t = new Tsr([2], [3, 5, 7])
        assert t.toString().contains("(2):[3.0, 5.0]")
        assert t.value64().length==2

        t = new Tsr(new int[]{2, 2}, new double[]{2, 4, 4})
        assert t.toString().contains("(2x2):[2.0, 4.0, 4.0, 2.0]")
        t = new Tsr(new int[]{2}, new double[]{3, 5, 7})
        assert t.toString().contains("(2):[3.0, 5.0]")
        assert t.value64().length==2
    }


    @Test
    void test_neureka_class()
    {
        Neureka.instance().reset()
        assert !Neureka.instance().settings().isLocked()
        assert !Neureka.instance().settings().indexing().legacy()
        assert !Neureka.instance().settings().debug().keepDerivativeTargetPayloads()
        assert Neureka.instance().settings().autoDiff().applyGradientWhenTensorIsUsed()
        Neureka.instance().settings().autoDiff().applyGradientWhenTensorIsUsed = false
        assert !Neureka.instance().settings().autoDiff().applyGradientWhenTensorIsUsed()
        assert Neureka.instance().settings().autoDiff().retainPendingErrorForJITProp()
        assert Neureka.version()=="0.0.2"//version
    }

    @Test
    void test_string_seeded_tensor()
    {
        Neureka.instance().reset()
        Tsr t1 = new Tsr([2, 3], "I am a seed! :)")
        Tsr t2 = new Tsr(new int[]{2, 3}, "I am a seed! :)")
        assert t1.toString()==t2.toString()
        Tsr t3 = new Tsr(new int[]{2, 3}, "I am also a seed! But different. :)")
        assert t1.toString()!=t3.toString()
    }

    @Test
    void test_indexing_after_reshaping()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)
        Tsr t1 = new Tsr([4, 3], 1..12)
        assert t1.i_of_idx(new int[]{2, 1})==7
        assert t1.i_of_idx(new int[]{1, 2})==5
        assert t1.idx_of_i(5)[0]==1
        assert t1.idx_of_i(5)[1]==2
        Tsr t2 = Function.create("[1, 0]:(I[0])").activate(t1)
        assert t2.i_of_idx(new int[]{1, 2})==7
        assert t2.idx_of_i(7)[0]==1
        assert t2.idx_of_i(7)[1]==3
        assert t1.toString().contains("[4x3]:(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)")
        assert t2.toString().contains("[3x4]:(1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0)")
        assert t1.i_of_idx(t1.idx_of_i(7)) == 7  // Element 7 '8.0' is at index 7!
        assert t2.i_of_idx(t2.idx_of_i(7)) == 10 // Element 7 '11.0' is at index 10!
    }

    @Test
    void test_reshaping()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)

        Function f = Function.create("[2, 0, 1]:(I[0])")
        Tsr t = new Tsr([3, 4, 2], 1..5)
        assert t.toString().contains("[3x4x2]:(1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0)")
        Tsr r = f.activate(t)
        assert r.toString().contains("[2x3x4]")
        assert r.toString().contains("[2x3x4]:(1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0)")
    }

    @Test
    void test_network_legacy_indexing()
    {
        Neureka.instance().settings().indexing().setLegacy(true)
        Neureka.instance().settings().view().setLegacy(true)

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
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(true)
        w_a * 3
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(false)
        assert w_a.toString().contains("g:(null)")
        assert !w_a.toString().contains("1.0, 3.0, 4.0, -1.0")
        assert !w_b.toString().contains("g:(null)")
        //TODO: calculate size and errors and check correctness!
    }

    @Test
    void test_network()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)
        Neureka.instance().settings().indexing().setLegacy(false)

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
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(true)
        w_a * 3
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(false)
        assert w_a.toString().contains("g:(null)")
        assert !w_a.toString().contains("1.0, 3.0, 4.0, -1.0")
        assert !w_b.toString().contains("g:(null)")
        //TODO: calculate size and errors and check correctness!
    }


    @Test
    void test_NN_with_manual_backprop()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)
        Device device = new DummyDevice()
        _testNN(device)

        //=========================================================================
        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
        //=========================================================================

        Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0)
        _testNN(gpu)

        //SOme more asserts:

        Tsr t = new Tsr([2, 2], 4).setRqsGradient(true).add(gpu)
        t.backward(1)
        Tsr g = t.find(Tsr.class)
        assert g.toString().contains("[2x2]:(1.0, 1.0, 1.0, 1.0)")
        assert t.toString().contains("[2x2]:(4.0, 4.0, 4.0, 4.0):g:(1.0, 1.0, 1.0, 1.0)")
        assert t.isOutsourced()
        assert g.isOutsourced()
        //t.setIsOutsourced(false)
        //assert !g.isOutsourced()

    }

    void _testNN(Device device)
    {
        Neureka.instance().settings().indexing().setLegacy(false)
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
        //println('Input:'+ X)
        //println('Expected output:'+ y)
        //println('Predicted output:'+ output)

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

    static Tsr sigmoid(Tsr x) {
        return new Tsr(x, "sig(I[0])")
        //return new Tsr(((Tsr.Create.E(x.shape())**(-x))+1), "1/I[0]")
        //return 1.0 / (1 + Tsr.Create.E(x.shape())**(-x))
    }

    static Tsr sigmoid_derivative(Tsr x) {
        return x * (-x + 1)
    }

    static void feedforward(Tsr weights1, Tsr weights2, Tsr input, Tsr output, Tsr layer1) {
        Tsr in0 = new Tsr([input, weights1], "i0xi1")
        layer1[] = sigmoid(in0)
        //println(layer1.toString("shp")+"=sig(  I"+input.toString("shp")+" X W"+weights1.toString("shp")+" )")
        Tsr in1 = new Tsr([layer1, weights2], "i0xi1")
        output[] = sigmoid(in1)
        //println(output.toString("shp")+"=sig( L1"+layer1.toString("shp")+" X W"+weights2.toString("shp")+" )\n")
    }

    static void backprop(Tsr weights1, Tsr weights2, Tsr input, Tsr output, Tsr layer1, Tsr y) {
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



    @Test
    void test_no_preemptive_apply_when_JIT_prop()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)
        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(true)
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(true)
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false)

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-3)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s = (a+b) * c // (2 - 3) * 3 = -3
        Tsr x = (s/a)+s // (-3)^2 -3 = 6

        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        x.backward(1)
        assert a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert c.has(JITProp.class)
        assert a.toString().contains("g:(0.75)")
        assert c.toString().contains("g:(null)")
        assert x.toString().contains("(-4.5)")

        def f = FunctionBuilder.build("I[0]*I[1]", false)
        Tsr[] inputs = new Tsr[2];
        inputs[0] = c
        inputs[1] = a
        Tsr result = f.activate(inputs)
        assert a.toString().contains("g:(0.75)")
        assert c.toString().contains("g:(null)")
        assert x.toString().contains("(-4.5)")

        f = FunctionBuilder.build("I[0]*I[1]", true)
        result = f.activate(inputs)
        assert a.toString().contains("g:(null)")
        assert c.toString().contains("g:(null)")
        assert x.toString().contains("(-4.5)")

        Neureka.instance().reset()
    }

    @Test
    void test_autograd_without_JIT_and_auto_apply()
    {
        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(false)
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(false)
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false)
        Neureka.instance().settings().view().setLegacy(true);

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-3)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s = (a+b) * c // (2 - 3) * 3 = -3
        Tsr x = (s/a)+s // (-3)^2 -3 = 6

        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        x.backward(1)
        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        assert a.toString().contains("g:(5.25)")// This has been checked!
        assert c.toString().contains("g:(-1.5)")// This has been checked!
        assert x.toString().contains("(-4.5)")
        a.applyGradient()
        c.applyGradient()
        assert a.toString().contains("(7.25):g:(null)")
        assert c.toString().contains("(1.5):g:(null)")
        Neureka.instance().reset()
    }


    @Test
    void test_indifferential_and_JIT_with_auto_apply()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(true)
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(true)
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false)
        Neureka.instance().settings().view().setLegacy(true)

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-3)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s = (a+b) * c // (2 - 3) * 3 = -3
        Tsr x = (s^a)+s // (-3)^2 -3 = 6

        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        x.backward(3)
        assert a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert c.has(JITProp.class)
        assert a.toString().contains("g:(NaN)")// NaN is expected! (derivative not possible!)
        assert c.toString().contains("g:(null)")
        Neureka.instance().reset()
    }

    @Test
    void test_no_JIT_prop_when_forward_AD(){
        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(true)
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(true)
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false)
        Neureka.instance().settings().view().setLegacy(true)

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)
        Tsr s = (a+b) * c
        Tsr x = (s^2)+s
        assert s.toString().contains("->d[1]:(-2.0)")
        assert s.toString().contains("->d[1]:(3.0)")
        assert s.toString().contains("[1]:(-6.0)")
        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        x.backward(3)
        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        assert a.toString().contains("g:(-99.0)")
        assert c.toString().contains("g:(66.0)")
        Neureka.instance().reset()
    }

    @Test
    void test_index_mapping(){

        Neureka.instance().settings().indexing().setLegacy(false)
        Neureka.instance().settings().view().setLegacy(true)
        Tsr t = new Tsr([3, 4],[
                1, 2, 3, 4,
                9, 8, 6, 5,
                4, 5, 6, 7
        ])
        t.label([
                ["1", "2", "3"],
                ["a", "b", "y", "z"]
        ])
        Tsr x = t["2", 1..2]
        assert x in t
        assert t.contains(x)
        assert x.toString().contains("[1x2]:(8.0, 6.0)")

        x = t["2".."3", "b".."y"]
        assert x in t
        assert t.contains(x)
        assert x.toString().contains("[2x2]:(8.0, 6.0, 5.0, 6.0)")

        t = new Tsr([2, 3, 4], 7)
        t.label([
                ["1", "2"],
                ["a", "b", "y"],
                ["tim", "tom", "tina", "tanya"]
        ])

        x = t["2", "b".."y", [["tim","tanya"]:2]]
        assert x in t
        assert t.contains(x)
        assert x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
        assert x.isVirtual()
        assert x.isSlice()
        assert t.isSliceParent()

        x = t["2", [["b".."y"]:1, ["tim","tanya"]:2]]
        assert x in t
        assert t.contains(x)
        assert x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
        assert x.isVirtual()
        assert x.isSlice()
        assert t.isSliceParent()

        assert t.sliceCount()==2

        x = t[[["2"]:1, ["b".."y"]:1, ["tim","tanya"]:2]]
        assert x in t
        assert t.contains(x)
        assert x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
        assert x.isVirtual()
        assert x.isSlice()
        assert t.isSliceParent()
        assert t.sliceCount()==3

        t.label(
            new String[][]{
                    new String[]{"1", "2"},
                    new String[]{"a", "b", "y"},
                    new String[]{"tim", "tom", "tina", "tanya"}
            }
        )

        x = t[["1","2"], "b".."y", [["tim","tanya"]:2]]
        assert x in t
        assert t.contains(x)
        assert x.toString().contains("[2x2x2]:(7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0)")
        assert x.isVirtual()
        assert x.isSlice()
        assert t.isSliceParent()
        assert t.sliceCount()==4

        WeakReference weak = new WeakReference(x)
        x = null
        System.gc()
        for(int i : 1..100){
            if(weak.get()==null) break
            Thread.sleep(10)
        }
        assert weak.get()!=null

    }

    @Test
    void test_indexing_modes(){
        Neureka.instance().settings().indexing().setLegacy(false)
        Tsr t0 = new Tsr([3, 2, 1], [
                1, 2,
                3, 4,
                5, 6
        ])
        Tsr x0 = new Tsr([1, 2, 3], [
                1, 2, 3,
                4, 5, 6
        ])
        /*
                9   12  15
                19  26  33
                29  40  51
         */
        Tsr out0 = new Tsr([t0, x0], "i0xi1")
        //TODO: asserts!
        //println(t0)
        //println(t0.value64(4))

        Neureka.instance().settings().indexing().setLegacy(true)
        Tsr t1 = new Tsr([3, 2, 1], [
                1, 2, 3,
                4, 5, 6
        ])
        Tsr x1 = new Tsr([1, 2, 3], [
                1, 2,
                3, 4,
                5, 6
        ])
        /*
                9   12  15
                19  26  33
                26  40  51
         */
        Tsr out1 = new Tsr([t0, x0], "i0xi1")
        //TODO: asserts!
        //println(t0.value64(4))
        //println(t1)

    }

    @Test
    void test_random_tensor(){
        Tsr r = new Tsr([2, 2], "jnrejn")
        assert r.toString().contains("0.02600E0, -2.06129E0, -0.48373E0, 0.94884E0")
        r = new Tsr([2, 2], "jnrejn2")
        assert !r.toString().contains("0.02600E0, -2.06129E0, -0.48373E0, 0.94884E0")
    }

    @Test
    void test_tensor_transpose()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)
        Neureka.instance().settings().indexing().setLegacy(true)
        Tsr t = new Tsr([2, 3], [
                1, 2,
                3, 4,
                5, 6
        ])
        t = t.T()
        assert t.toString().contains("[3x2]:(1.0, 3.0, 5.0, 2.0, 4.0, 6.0)")
        Neureka.instance().settings().indexing().setLegacy(false)
        t = new Tsr([2, 3], [
                1, 2, 3,
                4, 5, 6
        ])
        t = t.T()
        assert t.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")

    }

    @Test
    void test_data_types()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)

        Tsr x = new Tsr(3)
        x.to32()
        assert x.getValue() instanceof float[]
        assert x.is32()
        assert x.value32(0)==3.0f

        x.to64()
        assert x.getValue() instanceof double[]
        assert x.is64()
        assert x.value32(0)==3.0f

        float[] value32 = new float[1]
        value32[0] = 5
        x.setValue(value32)
        assert x.getValue() instanceof float[]
        assert x.is32()
        assert x.value32(0)==5.0f

        double[] value64 = new double[1]
        value64[0] = 4.0
        x.setValue(value64)
        assert x.getValue() instanceof double[]
        assert x.is64()
        assert x.value32(0)==4.0f

        assert x.isLeave()
        assert !x.isBranch()
        assert !x.isOutsourced()
        assert !x.isVirtual()
        assert !x.isSlice()
        assert !x.isSliceParent()
        assert !x.belongsToGraph()
        assert x.device() !=null
        assert x.device() instanceof HostCPU
        assert x.rank()==1
        assert !x.rqsGradient()
        assert x.size()==1

        Tsr y = new Tsr(x, "th(I[0])")
        assert y.isBranch()
        assert !y.isLeave()
        assert y.belongsToGraph()
        assert x.belongsToGraph()
        assert y.toString().contains("[1]:(0.97014E0)")

        int[] shape = new int[1]
        shape[0] = 4
        x = Tsr.Create.newRandom(shape)
        assert  x.toString().contains("[4]:(-0.14690E0, -0.13858E0, -2.30775E0, 0.67281E0)")
        x = Tsr.Create.newRandom(shape, 106605040595L)
        assert x.toString().contains("[4]:(-0.36765E0, -0.45818E0, -1.6556E0, 0.73242E0)")
    }

    @Test
    void test_power_overloaded_aAnd_input_functions()
    {
        Tsr x = new Tsr(3).setRqsGradient(true)
        Neureka.instance().settings().view().setLegacy(true)
        Tsr b = new Tsr(-4)
        Tsr w = new Tsr(2)
        Tsr y = ((x+b)*w)**2
        assert y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")
        y = ((x+b)*w)^2
        assert y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(true);
        y.backward(new Tsr(1))
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false);
        assert new Tsr([y], "Ig[0]").toString().equals("empty")
        assert new Tsr([x], "Ig[0]").toString().equals("empty")
        Tsr[] trs = new Tsr[1]
        trs[0] = x
        assert FunctionBuilder.build("Ig[0]", false).activate(trs).toString().equals("[1]:(-8.0)")
        trs[0] = y
        assert FunctionBuilder.build("Ig[0]", false).activate(trs).toString().contains("[1]:(4.0); ->d[1]:(-8.0)")
    }

    @Test
    void test_tensor_manipulation_and_IO()
    {
        Neureka.instance().settings().indexing().setLegacy(true)//TODO: repeat tests with default indexing

        Tsr t = new Tsr([2, 2], [
                1.0, 4.0,
                2.0, 7.0,
        ])
        Tsr v = new Tsr([2, 2], [1.0, -1.0, 1.0, -1.0])
        Tsr.IO.addInto(t, v)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 3.0, 6.0)")

        Tsr.IO.addInto(t, 2, 3.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 6.0, 6.0)")

        int[] idx = new int[2]
        idx[1] = 1
        Tsr.IO.addInto(t, idx, -9.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 6.0)")
        assert Tsr.IO.getFrom(t, idx)==-3.0

        idx[0] = 1
        Tsr.IO.mulInto(t, idx, -1)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, -6.0)")

        Tsr.IO.mulInto(t, 3, -2)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 12.0)")

        Tsr.IO.setInto(t, idx, 0.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 0.0)")

        Tsr.IO.setInto(t, 2, 99.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 99.0, 0.0)")
        //---
        Tsr.IO.subInto(t, 2, 99.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 0.0, 0.0)")
        idx[0] = 0
        Tsr.IO.subInto(t, idx, -9.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 9.0, 0.0)")

        Tsr.IO.subInto(t, new Tsr([2, 2], [1, 2, 3, 4]))
        assert t.toString().contains("[2x2]:(1.0, 1.0, 6.0, -4.0)")

        Neureka.instance().settings().indexing().setLegacy(false)

    }

    @Test
    void test_operator_overloading_of_SDK_types()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy true
        part_1 :
        {
            Tsr a = new Tsr(2).setRqsGradient(true)
            Tsr b = new Tsr(-4)
            Tsr c = new Tsr(3).setRqsGradient(true)
            assert (a/a).toString().contains("[1]:(1.0)")
            assert (c%a).toString().contains("[1]:(1.0)")
            assert (((b/b)^c%a)*3).toString().contains("[1]:(3.0)")
            assert (a *= b).toString().contains("(-8.0)")
            assert (a += -c).toString().contains("(-11.0)")
            assert (a -= c).toString().contains("(-14.0)")
            assert (a /= new Tsr(2)).toString().contains("(-7.0)")
            assert (a %= c).toString().contains("(-1.0)")
        }
        part_2 :
        {
            Tsr a = new Tsr(5)
            Tsr b = new Tsr(3)

            assert (2+a).toString().contains("7.0")
            assert (2*b).toString().contains("6.0")
            assert (6/b).toString().contains("2.0")
            assert (2^b).toString().contains("8.0")
            assert (2**b).toString().contains("8.0")
            assert (4-a).toString().contains("-1.0")

            assert (2.0+a).toString().contains("7.0")
            assert (2.0*b).toString().contains("6.0")
            assert (6.0/b).toString().contains("2.0")
            assert (2.0^b).toString().contains("8.0")
            assert (2.0**b).toString().contains("8.0")
            assert (4.0-a).toString().contains("-1.0")
        }
    }

    @Test
    void test_pending_error_optimization()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().legacy = true

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s =  (a*b) + 2
        Tsr x = s * (s+c)

        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(false)
        x.backward(new Tsr(1))
        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(true)
        assert c.toString().contains("(3.0):g:(-6.0)")
        assert a.toString().contains("(2.0):g:(36.0)")

        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(false)
        x.backward(4)
        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(true)
        assert c.toString().contains("(3.0):g:(-6.0)")
        assert a.toString().contains("(2.0):g:(36.0)")
    }

    @Test
    void test_pending_error_optimization_2()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s =  (a*b) + 2
        Tsr x = s * (s+c)

        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(false)
        x.backward(1)
        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(true)
        assert c.toString().contains("(3.0):g:(-6.0)")
        assert a.toString().contains("(2.0):g:(36.0)")

        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(true)
        x.backward(new Tsr(4))
        Neureka.instance().settings().autoDiff().setRetainPendingErrorForJITProp(true)
        assert c.toString().contains("(3.0):g:(-6.0)")
        assert a.toString().contains("(2.0):g:(36.0)")
    }


    @Test
    void test_JIT_propagation_variant_one()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)
        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s =  (a*b) + 2
        Tsr x = s * (s+c)

        x.backward(new Tsr(1))
        assert c.toString().contains("g:(-6.0)")
        assert a.toString().contains("g:(null)")
        a.applyGradient()
        assert c.toString().contains("g:(-6.0)")
        assert a.toString().contains("(38.0):g:(null)")
        //---
    }

    @Test
    void test_JIT_propagation_variant_two()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setLegacy(true)
        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s =  (a*b) + 2 // -6 = (2*-4) +2
        Tsr x = s * (s+c) //  -6 * (-6+3) // 18

        x.backward(new Tsr(1))

        assert c.toString().contains("g:(-6.0)")
        assert a.toString().contains("g:(null)")
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(true)
        Tsr y = a+3 //JIT-prop will be activated here...
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(false)
        assert y.toString().contains("(41.0)")
        assert c.toString().contains("g:(-6.0)")
        assert a.toString().contains("(38.0):g:(null)")
    }

    @Test
    void test_adding_device_to_tensor()
    {
        Neureka.instance().reset()
        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
        Device gpu = Device.find("nvidia")
        def t = new Tsr([3, 4, 1], 3).add(gpu)
        assert gpu.has(t)
    }

}
