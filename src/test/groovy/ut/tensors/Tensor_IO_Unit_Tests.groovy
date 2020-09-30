package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.acceleration.host.HostCPU
import neureka.calculus.Function
import spock.lang.Specification

class Tensor_IO_Unit_Tests extends  Specification
{
    def 'Tensors can be instantiated with String seed.'()
    {
        given : 'The current Neureka instance configuration is being reset.'
            Neureka.instance().reset()
        and : 'Three seeded 2D tensors are being instantiated.'
            Tsr t1 = new Tsr([2, 3], "I am a seed! :)")
            Tsr t2 = new Tsr(new int[]{2, 3}, "I am a seed! :)")
            Tsr t3 = new Tsr(new int[]{2, 3}, "I am also a seed! But different. :)")
        expect : 'Equal seeds produce equal values.'
            assert t1.toString()==t2.toString()
            assert t1.toString()!=t3.toString()
    }

    def 'Smart tensor constructors yield expected results.'()
    {
        given : 'Three scalar tensors.'
            Tsr a = new Tsr(3)
            Tsr b = new Tsr(2)
            Tsr c = new Tsr(-1)

        when : Tsr t = new Tsr("1+", a, "*", b)
        then : assert t.toString().contains("7.0")
        when : t = new Tsr("1", "+", a, "*", b)
        then : assert t.toString().contains("7.0")
        when : t = new Tsr("(","1+", a,")", "*", b)
        then : assert t.toString().contains("8.0")
        when : t = new Tsr("(","1", "+", a,")", "*", b)
        then : assert t.toString().contains("8.0")
        when : t = new Tsr("(", c, "*3)+", "(","1+", a,")", "*", b)
        then : assert t.toString().contains("5.0")
        when : t = new Tsr("(", c, "*","3)+", "(","1+", a,")", "*", b)
        then : assert t.toString().contains("5.0")
        when : t = new Tsr("(", c, "*","3", ")+", "(","1+", a,")", "*", b)
        then : assert t.toString().contains("5.0")

        when : t = new Tsr([2, 2], [2, 4, 4])
        then : assert t.toString().contains("(2x2):[2.0, 4.0, 4.0, 2.0]")
        when : t = new Tsr([2], [3, 5, 7])
        then :
            assert t.toString().contains("(2):[3.0, 5.0]")
            assert t.value64().length==2

        // Now the same with primitive array ! :
        when : t = new Tsr(new int[]{2, 2}, new double[]{2, 4, 4})
        then : assert t.toString().contains("(2x2):[2.0, 4.0, 4.0, 2.0]")
        when : t = new Tsr(new int[]{2}, new double[]{3, 5, 7})
        then :
            assert t.toString().contains("(2):[3.0, 5.0]")
            assert t.value64().length==2

    }


    def 'Indexing after reshaping works as expected.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().view().isUsingLegacyView = true

        and :
            Tsr t1 = new Tsr([4, 3], 1..12)

        when :

            def t1_ioi_1 = t1.i_of_idx(new int[]{2, 1})
            def t1_ioi_2 = t1.i_of_idx(new int[]{1, 2})
            def t1_idx   = t1.idx_of_i(5)

            Tsr t2 = Function.create(" [ 1, 0 ]:( I[0] ) ")(t1)
            def t2_ioi_1 = t2.i_of_idx(new int[]{1, 2})
            def t2_idx = t2.idx_of_i(7)

            def t1_ioi_3 = t1.i_of_idx(t1.idx_of_i(7)) // Element 7 '8.0' is at index 7!
            def t2_ioi_2 =  t2.i_of_idx(t2.idx_of_i(7)) // Element 7 '11.0' is at index 10!

        then :
            t1_ioi_1 == 7
            t1_ioi_2 == 5
            t1_idx[0] == 1
            t1_idx[1] == 2

            t2_ioi_1 == 7
            t2_idx[0] == 1
            t2_idx[1] == 3

            t1_ioi_3 == 7 // Element 7 '8.0' is at index 7!
            t2_ioi_2 == 10 // Element 7 '11.0' is at index 10!

            t1.toString().contains("[4x3]:(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)")
            t2.toString().contains("[3x4]:(1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0)")
    }


    def 'Passing String seed to tensor produces expected values.'()
    {
        when : Tsr r = new Tsr([2, 2], "jnrejn")
        then : r.toString().contains("0.02600E0, -2.06129E0, -0.48373E0, 0.94884E0")
        when : r = new Tsr([2, 2], "jnrejn2")
        then : !r.toString().contains("0.02600E0, -2.06129E0, -0.48373E0, 0.94884E0")
    }

    def 'Tensor value type can not be changed by passing float or double arrays to it.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            Tsr x = new Tsr(3)

        when :
            float[] value32 = new float[1]
            value32[0] = 5
            x.setValue(value32)

        then :
            !(x.getValue() instanceof float[])
            !x.is32()
            x.value32(0)==5.0f
            x.value64(0)==5.0d

        when :
            double[] value64 = new double[1]
            value64[0] = 4.0
            x.setValue(value64)

        then :
            x.getValue() instanceof double[]
            x.is64()
            x.value32(0)==4.0f
            x.value64(0)==4.0d

            x.isLeave()
            !x.isBranch()
            !x.isOutsourced()
            !x.isVirtual()
            !x.isSlice()
            !x.isSliceParent()
            !x.belongsToGraph()
            x.device() !=null
            x.device() instanceof HostCPU
            x.rank()==1
            !x.rqsGradient()
            x.size()==1

        when : x.to32()
        then : x.value instanceof float[]

        when :
            value64 = new double[1]
            value64[0] = 7.0
            x.setValue(value64)

        then :
            !(x.getValue() instanceof double[])
            !x.is64()
            x.value32(0)==7.0f
            x.value64(0)==7.0d

    }


    def 'Tensors value type can be changed by calling "to64()" and "to32()".'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            Tsr x = new Tsr(3)

        when : x.to32()
        then :
            assert x.getValue() instanceof float[]
            assert x.is32()
            assert x.value32(0)==3.0f

        when : x.to64()
        then :
            assert x.getValue() instanceof double[]
            assert x.is64()
            assert x.value32(0)==3.0f
    }


    def 'A tensor produced by a function has expected properties.'()
    {
        given:
            Neureka.instance().reset()
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            Tsr x = new Tsr(4)

        when: Tsr y = new Tsr(x, "th(I[0])")

        then:
            y.isBranch()
            !y.isLeave()
            y.belongsToGraph()
            x.belongsToGraph()
            y.toString().contains("[1]:(0.97014E0)")
    }

    def 'A tensor produced by the static "Tsr.Create.newRandom(shape)" has expected "random" value.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().view().isUsingLegacyView = true

        when :
            int[] shape = new int[1]
            shape[0] = 4
            Tsr x = Tsr.Create.newRandom(shape)

        then : assert  x.toString().contains("[4]:(-0.14690E0, -0.13858E0, -2.30775E0, 0.67281E0)")
        when : x = Tsr.Create.newRandom(shape, 106605040595L)
        then : assert x.toString().contains("[4]:(-0.36765E0, -0.45818E0, -1.6556E0, 0.73242E0)")
    }


    void 'Tensor values can be manipulated via static method calls within the "Tsr.IO" class.'()
    {
        given :
        Neureka.instance().reset()
        Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true)//TODO: repeat tests with default indexing
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenRequested(false)
        Neureka.instance().settings().view().setIsUsingLegacyView(true)
        Tsr t = new Tsr([2, 2], [
                1.0, 4.0,
                2.0, 7.0,
        ])
        Tsr v = new Tsr([2, 2], [1.0, -1.0, 1.0, -1.0])

        when : Tsr.IO.addInto(t, v)
        then : t.toString().contains("[2x2]:(2.0, 3.0, 3.0, 6.0)")

        when : Tsr.IO.addInto(t, 2, 3.0)
        then : t.toString().contains("[2x2]:(2.0, 3.0, 6.0, 6.0)")

        when :
            int[] idx = new int[2]
            idx[1] = 1
            Tsr.IO.addInto(t, idx, -9.0)
        then :
            t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 6.0)")
            Tsr.IO.getFrom(t, idx)==-3.0d

        when :
            idx[0] = 1
            Tsr.IO.mulInto(t, idx, -1)

        then : t.toString().contains("[2x2]:(2.0, 3.0, -3.0, -6.0)")

        when : Tsr.IO.mulInto(t, 3, -2)
        then : t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 12.0)")

        when : Tsr.IO.setInto(t, idx, 0.0)
        then : t.toString().contains("[2x2]:(2.0, 3.0, -3.0, 0.0)")

        when : Tsr.IO.setInto(t, 2, 99.0)
        then : t.toString().contains("[2x2]:(2.0, 3.0, 99.0, 0.0)")

        when : Tsr.IO.subInto(t, 2, 99.0)
        then : t.toString().contains("[2x2]:(2.0, 3.0, 0.0, 0.0)")

        when :
            idx[0] = 0
            Tsr.IO.subInto(t, idx, -9.0)
        then : t.toString().contains("[2x2]:(2.0, 3.0, 9.0, 0.0)")

        when : Tsr.IO.subInto(t, new Tsr([2, 2], [1, 2, 3, 4]))
        then : t.toString().contains("[2x2]:(1.0, 1.0, 6.0, -4.0)")

    }


    def 'Adding OpenCL device to tensor makes tensor be "outsourced" and contain the Device instance as component.'()
    {
        given : 'Neureka can access OpenCL (JOCL).'
            if ( !Neureka.instance().canAccessOpenCL() ) return
            Neureka.instance().reset()
            Device gpu = Device.find("nvidia")
            Tsr t = new Tsr([3, 4, 1], 3)

        expect : 'The following is to be expected with respect to the given :'
            !t.has(Device.class)
            !t.isOutsourced()
            !gpu.has(t)

        when : 'The tensor is being added to the OpenCL device...'
            t.add(gpu)

        then : 'The now "outsourced" tensor has a reference to the device and vice versa!'
            t.has(Device.class)
            t.isOutsourced()
            gpu.has(t)

    }

}
