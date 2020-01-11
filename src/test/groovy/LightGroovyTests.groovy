
import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.CPU
import neureka.acceleration.Device
import neureka.autograd.JITProp
import neureka.calculus.factory.assembly.FunctionBuilder
import org.junit.Test

class LightGroovyTests
{

    @Test
    void testNoJITPropWhenForwardAD(){
        Neureka.Settings.AD.setRetainPendingErrorForJITProp(true)
        Neureka.Settings.AD.setApplyGradientWhenTensorIsUsed(true)
        Neureka.Settings.AD.setRetainGraphDerivativesAfterBackward(false)
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
        Neureka.Settings.reset()
    }

    @Test
    void testIndexMapping(){

        Neureka.Settings.Indexing.setLegacy(false)
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
        assert x.toString().contains("[1x2]:(8.0, 6.0)")

        x = t["2".."3", "b".."y"]

        assert x.toString().contains("[2x2]:(8.0, 6.0, 5.0, 6.0)")

        t = new Tsr([2, 3, 4], 7)
        t.label([
                ["1", "2"],
                ["a", "b", "y"],
                ["tim", "tom", "tina", "tanya"]
        ])

        x = t["2", "b".."y", [["tim","tanya"]:2]]

        assert x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
        assert x.isVirtual()
        assert x.isSlice()
        assert t.isSliceParent()

        x = t["2", [["b".."y"]:1, ["tim","tanya"]:2]]

        assert x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
        assert x.isVirtual()
        assert x.isSlice()
        assert t.isSliceParent()

        assert t.sliceCount()==2

        x = t[[["2"]:1, ["b".."y"]:1, ["tim","tanya"]:2]]

        assert x.toString().contains("[1x2x2]:(7.0, 7.0, 7.0, 7.0)")
        assert x.isVirtual()
        assert x.isSlice()
        assert t.isSliceParent()

        assert t.sliceCount()==3

    }

    @Test
    void testIndexingModes(){
        Neureka.Settings.Indexing.setLegacy(false)
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
        println(t0)

        println(t0.value64(4))

        Neureka.Settings.Indexing.setLegacy(true)

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
        println(t0.value64(4))

        println(t1)

    }

    @Test
    void testRandomTensor(){
        Tsr r = new Tsr([2, 2], "jnrejn")
        assert r.toString().contains("7.205, -6.63625E0, -7.2275, 0.77874E0")
        r = new Tsr([2, 2], "jnrejn2")
        assert !r.toString().contains("7.205, -6.63625E0, -7.2275, 0.77874E0")
    }

    @Test
    void testTranspose(){
        Neureka.Settings.Indexing.setLegacy(true)
        Tsr t = new Tsr([2, 3], [
                1, 2,
                3, 4,
                5, 6
        ])
        t = t.T()
        assert t.toString().contains("[3x2]:(1.0, 3.0, 5.0, 2.0, 4.0, 6.0)")
        Neureka.Settings.Indexing.setLegacy(false)
        t = new Tsr([2, 3], [
                1, 2, 3,
                4, 5, 6
        ])
        t = t.T()
        assert t.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")

    }

    @Test
    void testDataTypes()
    {
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
        assert x.device() instanceof CPU
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
        x = Tsr.fcn.create.newRandom(shape)
        assert  x.toString().contains("[4]:(-0.1575, -1.57875E0, 5.2775, 0.40125)")
        x = Tsr.fcn.create.newRandom(shape, 106605040595L)
        assert x.toString().contains("[4]:(0.3675, -4.30875E0, -6.60625E0, 1.265E0)")
    }

    @Test
    void testPowerOverloadedAndInputFunctions()
    {
        Tsr x = new Tsr(3).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr w = new Tsr(2)
        Tsr y = ((x+b)*w)**2
        assert y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")
        y = ((x+b)*w)^2
        assert y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")
        Neureka.Settings.AD.setRetainGraphDerivativesAfterBackward(true);
        y.backward(new Tsr(1))
        Neureka.Settings.AD.setRetainGraphDerivativesAfterBackward(false);
        assert new Tsr([y], "Ig[0]").toString().equals("empty")
        assert new Tsr([x], "Ig[0]").toString().equals("empty")
        Tsr[] trs = new Tsr[1]
        trs[0] = x
        assert FunctionBuilder.build("Ig[0]", false).activate(trs).toString().equals("[1]:(-8.0)")
        trs[0] = y
        assert FunctionBuilder.build("Ig[0]", false).activate(trs).toString().contains("[1]:(4.0); ->d[1]:(-8.0)")
    }

    @Test
    void testTensorManipulation()
    {
        Neureka.Settings.Indexing.setLegacy(true)//TODO: repeat tests with default indexing

        Tsr t = new Tsr([2, 2], [
                1.0, 4.0,
                2.0, 7.0,
        ])
        Tsr v = new Tsr([2, 2], [1.0, -1.0, 1.0, -1.0])
        Tsr.fcn.io.addInto(t, v)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 3.0, 6.0)")

        Tsr.fcn.io.addInto(t, 2, 3.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 6.0, 6.0)")

        int[] idx = new int[2]
        idx[1] = 1
        Tsr.fcn.io.addInto(t, idx, -9.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 6.0)")
        assert Tsr.fcn.io.getFrom(t, idx)==-3.0

        idx[0] = 1
        Tsr.fcn.io.mulInto(t, idx, -1)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, -6.0)")

        Tsr.fcn.io.mulInto(t, 3, -2)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 12.0)")

        Tsr.fcn.io.setInto(t, idx, 0.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 0.0)")

        Tsr.fcn.io.setInto(t, 2, 99.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 99.0, 0.0)")
        //---
        Tsr.fcn.io.subInto(t, 2, 99.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 0.0, 0.0)")
        idx[0] = 0
        Tsr.fcn.io.subInto(t, idx, -9.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 9.0, 0.0)")

        Tsr.fcn.io.subInto(t, new Tsr([2, 2], [1, 2, 3, 4]))
        assert t.toString().contains("[2x2]:(1.0, 1.0, 6.0, -4.0)")

        Neureka.Settings.Indexing.setLegacy(false)

    }

    @Test
    void testOperations()
    {
        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)
        assert (a/a).toString().contains("[1]:(1.0)")
        assert (c%a).toString().contains("[1]:(1.0)")
        assert (((b/b)^c%a)*3).toString().contains("[1]:(3.0)")
    }

    @Test
    void testPendingErrorOptimization()
    {
        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s =  (a*b) + 2
        Tsr x = s * (s+c)

        Neureka.Settings.AD.setRetainPendingErrorForJITProp(false)
        x.backward(new Tsr(1))
        Neureka.Settings.AD.setRetainPendingErrorForJITProp(true)
        assert c.toString().contains("(3.0):g:(-6.0)")
        assert a.toString().contains("(2.0):g:(36.0)")

        Neureka.Settings.AD.setRetainPendingErrorForJITProp(false)
        x.backward(4)
        Neureka.Settings.AD.setRetainPendingErrorForJITProp(true)
        assert c.toString().contains("(3.0):g:(-6.0)")
        assert a.toString().contains("(2.0):g:(36.0)")
    }

    @Test
    void testPendingErrorOptimization2()
    {
        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s =  (a*b) + 2
        Tsr x = s * (s+c)

        Neureka.Settings.AD.setRetainPendingErrorForJITProp(false)
        x.backward(1)
        Neureka.Settings.AD.setRetainPendingErrorForJITProp(true)
        assert c.toString().contains("(3.0):g:(-6.0)")
        assert a.toString().contains("(2.0):g:(36.0)")

        Neureka.Settings.AD.setRetainPendingErrorForJITProp(true)
        x.backward(new Tsr(4))
        Neureka.Settings.AD.setRetainPendingErrorForJITProp(true)
        assert c.toString().contains("(3.0):g:(-6.0)")
        assert a.toString().contains("(2.0):g:(36.0)")
    }


    @Test
    void testJITPropagationVariantOne()
    {
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
    void testJITPropagationVariantTwo()
    {
        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s =  (a*b) + 2
        Tsr x = s * (s+c)

        x.backward(new Tsr(1))

        assert c.toString().contains("g:(-6.0)")
        assert a.toString().contains("g:(null)")
        Neureka.Settings.AD.setApplyGradientWhenTensorIsUsed(true)
        Tsr y = a+3 //JIT-prop will be activated here...
        Neureka.Settings.AD.setApplyGradientWhenTensorIsUsed(false)
        assert y.toString().contains("(41.0)")
        assert c.toString().contains("g:(-6.0)")
        assert a.toString().contains("(38.0):g:(null)")
    }

    @Test
    void testAddingDeviceToTensor()
    {
        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
        Device gpu = Neureka.findAcceleratorByName("nvidia")
        def t = new Tsr([3, 4, 1], 3).add(gpu)
        assert gpu.has(t)
    }

}
