package st


import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.HostCPU
import neureka.devices.opencl.OpenCLDevice
import neureka.devices.opencl.OpenCLPlatform
import neureka.utility.TsrAsString
import spock.lang.Specification

import testutility.mock.DummyDevice


class Cross_Device_Sliced_Tensor_System_Test extends Specification
{
    def setup() {
        Neureka.instance().reset()
        // Configure printing of tensors to be more compact:
        Neureka.instance().settings().view().asString = TsrAsString.configFromCode("dgc")
    }

    def 'Cross device sliced tensor integration test runs without errors.'(
            Device device, boolean legacyIndexing
    ) {
        given :
            if ( device == null ) return
            Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed = false
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            if ( device instanceof OpenCLDevice && !Neureka.instance().canAccessOpenCL() ) return

        when :
            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(legacyIndexing)
            if ( device instanceof OpenCLDevice ) OpenCLPlatform.PLATFORMS().get(0).recompile()

        Tsr x = new Tsr([1], 3).setRqsGradient(true)
        Tsr b = new Tsr([1], -4)
        Tsr w = new Tsr([1], 2)
        device.store(x).store(b).store(w)
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        Tsr y = new Tsr([x, b, w], "((i0+i1)*i2)^2")
        then:
        assert y.idxmap() != null
        assert y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when:
        y.backward(new Tsr(2))

        y = ((x+b)*w)**2

        then:
        assert y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when:
        y.backward(new Tsr(2))
        x.toString().contains("-32.0")
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

        device.store(a)
        b = a[[-1..-3, -6..-3]]
        def s = a[[1, -2]]

        then:
        assert s == ((legacyIndexing)?9.0:2.0)
        assert b.toString().contains(
                (legacyIndexing)
                        ?"2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"
                        :"7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
        )

        assert b.spread() != null

        when:
        b = a[-3..-1, 0..3]
        s = a[1, -2]

        then:
        assert s == ( (legacyIndexing) ? 9.0 : 2.0 )
        assert b.toString().contains(
                (legacyIndexing)
                        ?"2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"
                        :"7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
        )
        assert b.spread() != null
        /**
         * 2, 3, 4,
         * 6, 7, 8,
         * 1, 2, 3,
         * 5, 6, 7,
         */
        //---
        when:
        if( device instanceof DummyDevice ) {
            a.value64()[1] = a.value64()[1] * 6
            a.value64()[7] = a.value64()[7] * 2
        } else {
            Tsr k = new Tsr([4, 6], [
                    1, 6, 1, 1,
                    1, 1, 1, 2,
                    1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1
            ])
            device.store( k )
            a[] = a * k
        }

        then:
        assert b.toString().contains(
                (legacyIndexing)
                        ? "12.0, 3.0, 4.0, 6.0, 7.0, 16.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0"
                        : "7.0, 16.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
        )

        //assert x.toString().contains("-16.0")
        //---

        when:
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

        then:
        assert d.toString().toString().contains(
                (legacyIndexing)?
                        "9.0, 5.0, 7.0, " +
                                "11.0, 13.0, 18.0, " +
                                "0.0, 3.0, 5.0, " +
                                "8.0, 10.0, 9.0"
                        :"4.0, 18.0, 12.0, 6.0, "+
                        "10.0, 7.0, 5.0, 8.0, "+
                        "3.0, 5.0, 7.0, 6.0"
        )
        //---
        when:
        b = a[1..3, 2..4]
        then:
        assert b.toString().contains(
                (legacyIndexing)
                        ?"1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 1.0, 2.0"
                        :"9.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0"
        )

        assert b.spread() != null
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
        when:
        b = a[[[0..3]:2, [1..4]:2]]
        then:
        assert b.toString().contains(
                (legacyIndexing)
                        ?"5.0, 7.0, 4.0, 6.0"
                        :"12.0, 4.0, 5.0, 7.0"
        )

        assert b.spread() != null
        /**
         1, 12, 3, 4,
         5, 6, 7, 16, => 5,  7,
         9, 1, 2, 3,
         4, 5, 6, 7, => 4,  6,
         8, 9, 1, 2,
         3, 4, 5, 6

         NON LEGACY INDEXING :
         1, 12, 3, 4, 5, 6, => 12, 4,
         7, 16, 9, 1, 2, 3,
         4, 5, 6, 7, 8, 9,  => 5,  7,
         1, 2, 3, 4, 5, 6

         */
        //---
        when:
        Tsr p = new Tsr([2,2], [2, 55, 4, 7]).set((device instanceof DummyDevice)?null:device)
        Tsr u = new Tsr([2,2], [5, 2, 7, 34]).set((device instanceof DummyDevice)?null:device)

        p[] = u
        //tester.testContains(p.toString(), ["5.0, 2.0, 7.0, 34.0"], "Testing slicing")
        then:
        assert p.toString().contains("5.0, 2.0, 7.0, 34.0")

        //---
        when:
        a[[[0..3]:2, [1..4]:2]] = new Tsr([2, 2], [1, 2, 3, 4])
        then:
        assert b.toString().contains("1.0, 2.0, 3.0, 4.0")
        assert a.toString().contains(
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
        when:
        a[1..2, 1..2] = new Tsr([2, 2], [8, 8, 8, 8])
        then:
        assert b.toString().contains(
                (legacyIndexing)?
                        "1.0, 8.0, " +
                                "3.0, 4.0"
                        :"1.0, 2.0, "+
                        "8.0, 4.0"
        )
        assert a.toString().contains(
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
        )
        /**a:>>
         1, 12, 3, 4,
         1, 8, 8, 16,
         9, 8, 8, 3,
         3, 5, 4, 7,
         8, 9, 1, 2,
         3, 4, 5, 6

         NON LEGACY INDEXING :
         1.0, 1.0, 3.0, 2.0, 5.0, 6.0,
         7.0, 8.0, 8.0, 1.0, 2.0, 3.0,
         4.0, 8.0, 8.0, 4.0, 8.0, 9.0,
         1.0, 2.0, 3.0, 4.0, 5.0, 6.0
         */

        //---
        //b>>
        //      1, 8,
        //      3, 4
        when:
        b.setRqsGradient(true)
        c = new Tsr([2, 2], [
                -2, 3,//-2 + 24 + 3 + 8
                1, 2,
        ])
        device.store(b).store(c) // -2 + 6 + 8 + 8 = 22
        x = new Tsr(b, "x", c) // This test is important because it tests convolution on slices!
        then:
        assert x.toString().contains(
                (legacyIndexing)?
                        "[1x1]:(33.0); ->d[2x2]:(-2.0, 3.0, 1.0, 2.0)"
                        :"[1x1]:(20.0); ->d[2x2]:(-2.0, 3.0, 1.0, 2.0)"
        )

        where:
            device               | legacyIndexing
            Device.find('gpu')   | true
            Device.find('gpu')   | false
            HostCPU.instance()   | true
            HostCPU.instance()   | false
    }




}
