
import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.function.factory.assembly.FunctionBuilder
import neureka.gui.swing.AbstractSpaceMap
import neureka.gui.swing.GraphBoard
import neureka.gui.swing.SurfaceObject
import org.junit.Test

class LightGroovyTests
{

    @Test
    void testIndexingModes(){
        Neureka.settings.tsr.SET_LEGACY_INDEXING(false)
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

        Neureka.settings.tsr.SET_LEGACY_INDEXING(true)

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
        Neureka.settings.tsr.SET_LEGACY_INDEXING(true)
        Tsr t = new Tsr([2, 3], [
                1, 2,
                3, 4,
                5, 6
        ])
        t = t.T()
        assert t.toString().contains("[3x2]:(1.0, 3.0, 5.0, 2.0, 4.0, 6.0)")
        Neureka.settings.tsr.SET_LEGACY_INDEXING(false)
        t = new Tsr([2, 3], [
                1, 2, 3,
                4, 5, 6
        ])
        t = t.T()
        assert t.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")

    }

    @Test
    void testNN() {
        Neureka.settings.tsr.SET_LEGACY_INDEXING(false)

        Tsr X = new Tsr(// input data: 5 vectors in binary form
            [5, 3, 1],
            [
                0, 0, 1,
                1, 1, 0,
                1, 0, 1,
                0, 1, 1,
                1, 1, 1
            ]
        )

        Tsr y = new Tsr(// output values (labels)
                [5, 1, 1],[0,1,1,1,0]
        )// [1, 5, 1],(0,1,1,1,0)

        Tsr input = X
        Tsr weights1 = new Tsr([1, input.shape()[1], 4],
                [4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01,
                 1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01,
                 3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01]
        )
        /*
            [1x5x4]:(...)
            w1 (3, 4) :
            [[4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01]
             [1.46755891e-01 9.23385948e-02 1.86260211e-01 3.45560727e-01]
             [3.96767474e-01 5.38816734e-01 4.19194514e-01 6.85219500e-01]]
        */
        Tsr weights2 = new Tsr([1, 1, 4], [0.20445225, 0.87811744, 0.02738759, 0.67046751])
        /*
            [1x1x4]:...
            w2 (4, 1) :
            [[0.20445225]
             [0.87811744]
             [0.02738759]
             [0.67046751]]
         */
        Tsr output = new Tsr(y.shape(), [0.0, 0.0, 0.0, 0.0, 0.0])
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
        assert !x.hasDataParent()
        assert !x.belongsToGraph()
        assert x.device()==null
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
        Neureka.settings.ad.RETAIN_GRAPH_DERIVATIVES_AFTER_BACKWARD = true
        y.backward(new Tsr(1))
        Neureka.settings.ad.RETAIN_GRAPH_DERIVATIVES_AFTER_BACKWARD = false
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
        Neureka.settings.tsr.SET_LEGACY_INDEXING(true)//TODO: repeat tests with default indexing

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

        Neureka.settings.tsr.SET_LEGACY_INDEXING(false)

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

        Neureka.settings.ad.RETAIN_PENDING_ERROR_FOR_JITPROP = false
        x.backward(new Tsr(1))
        Neureka.settings.ad.RETAIN_PENDING_ERROR_FOR_JITPROP = true
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
        Neureka.settings.ad.APPLY_GRADIENT_WHEN_TENSOR_IS_USED = true
        Tsr y = a+3 //JIT-prop will be activated here...
        Neureka.settings.ad.APPLY_GRADIENT_WHEN_TENSOR_IS_USED = false
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
