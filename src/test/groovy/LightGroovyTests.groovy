
import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.acceleration.opencl.OpenCLPlatform
import neureka.acceleration.opencl.utility.DeviceQuery
import org.junit.Test
import util.DummyDevice
import util.NTester
import util.NTester_Tensor

class LightGroovyTests {

    @Test
    void testDataTypes(){

        Tsr x = new Tsr(3)

        x.to32()
        assert x.getValue() instanceof float[]
        assert x.is32()

        x.to64()
        assert x.getValue() instanceof double[]
        assert x.is64()

        float[] value32 = new float[1]
        x.setValue(value32)
        assert x.getValue() instanceof float[]
        assert x.is32()

        double[] value64 = new double[1]
        x.setValue(value64)
        assert x.getValue() instanceof double[]
        assert x.is64()

        assert x.isLeave()
        assert !x.isOutsourced()
        assert !x.isVirtual()
        assert !x.hasDataParent()
        assert !x.belongsToGraph()
        assert x.device()==null
        assert x.rank()==1
        assert !x.rqsGradient()
        assert x.size()==1

        int[] shape = new int[1]
        shape[0] = 4
        x = Tsr.fcn.create.newRandom(shape)
        assert  x.toString().contains("[4]:(-0.1575, -1.57875E0, 5.2775, 0.40125)")
        x = Tsr.fcn.create.newRandom(shape, 106605040595L)
        assert x.toString().contains("[4]:(0.3675, -4.30875E0, -6.60625E0, 1.265E0)")

    }

    @Test
    void testPowerOverloaded(){
        Tsr x = new Tsr(3).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr w = new Tsr(2)
        Tsr y = ((x+b)*w)**2
        // y: "[1]:(4.0); ->d[1]:(-8.0), "
        assert y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")
    }

    @Test
    void testTensorManipulation(){

        Tsr t = new Tsr([2, 2], [
                1.0, 4.0,
                2.0, 7.0,
        ])
        Tsr v = new Tsr([2, 2], [1.0, -1.0, 1.0, -1.0])
        Tsr.fcn.io.addInto(t, v)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 3.0, 6.0)")

        Tsr.fcn.io.addInto(t, 2, 3.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 6.0, 6.0)")

        //
        int[] idx = new int[2]
        idx[1] = 1
        Tsr.fcn.io.addInto(t, idx, -9.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 6.0)")

        assert Tsr.fcn.io.getFrom(t, idx)==-3.0

        idx[0] = 1
        Tsr.fcn.io.mulInto(t, idx, -1)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, -6.0)")

        Tsr.fcn.io.setInto(t, idx, 0.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 0.0)")

        Tsr.fcn.io.setInto(t, 2, 99.0)
        assert t.toString().contains("[2x2]:(2.0, 3.0, 99.0, 0.0)")


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

        x.backward(new Tsr(1))
        assert c.toString().contains("g:(-6.0)")
        assert a.toString().contains("g:(36.0)")
        println(x)
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
