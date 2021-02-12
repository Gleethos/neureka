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
        Neureka.instance().settings().view().asString = "dgc"
    }

    def 'Cross device sliced tensor integration test runs without errors.'(
            Device device
    ) {
        given :
            if ( device == null ) return
            Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed = false
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            if ( device instanceof OpenCLDevice && !Neureka.instance().canAccessOpenCL() ) return

        when :
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
            y.indicesMap() != null
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when:
            y.backward(new Tsr(2))
            y = ((x+b)*w)**2

        then:
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when:
            y.backward(new Tsr(2))
            x.toString().contains("-32.0")
            y = b + w * x

            /**
             *  Subset:
             */
            Tsr a = new Tsr([4, 6], [
                    1, 2, 3, 4, 5, 6,
                    7, 8, 9, 1, 2, 3,
                    4, 5, 6, 7, 8, 9,
                    1, 2, 3, 4, 5, 6
            ])
            /*
                1, 2, 3, 4, 5, 6,
                7, 8, 9, 1, 2, 3, => 7, 8, 9, 1
                4, 5, 6, 7, 8, 9, => 4, 5, 6, 7
                1, 2, 3, 4, 5, 6  => 1, 2, 3, 4
             */

            device.store(a)
            b = a[[-1..-3, -6..-3]]
            def s = a[[1, -2]]

        then:
            s.toString() == "[1x1]:(2.0)"
            s.getValueAt(0) == 2.0
            b.toString().contains(
                    "7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
            )
            b.spread() != null

        when:
            b = a[-3..-1, 0..3]
            s = a[1, -2]

        then:
            s.toString() == "[1x1]:(2.0)"
            s.getValueAt(0) == 2.0
            s.getDataAt(0) == 1.0
            s.getDataAt(1) == 2.0
            b.toString().contains(
                    "7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
            )
            b.spread() != null
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
            b.toString().contains(
                    "7.0, 16.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0"
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
            (d.NDConf.asInlineArray() as List) == ( [3, 4, 4, 1, 4, 1, 0, 0, 1, 1] )
            (b.NDConf.asInlineArray() as List) == ( [3, 4, 6, 1, 4, 1, 1, 0, 1, 1] )
            (c.NDConf.asInlineArray() as List) == ( [3, 4, 4, 1, 4, 1, 0, 0, 1, 1] )


            d.toString().contains(
                "4.0, 18.0, 12.0, 6.0, "+
                        "10.0, 7.0, 5.0, 8.0, "+
                        "3.0, 5.0, 7.0, 6.0"
            ) // 9.0, 5.0, 7.0, 10.0, 12.0, 9.0, 15.0, 10.0, 3.0, 5.0, 7.0, 6.0
              // 9.0, 5.0, 7.0, 10.0, 12.0, 9.0, 15.0, 10.0, 3.0, 5.0, 7.0, 6.0
            //---
        when:
            b = a[1..3, 2..4]
        then:
            b.toString().contains(
                    "9.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0"
            )

            b.spread() != null
            /**
             1, 12, 3, 4, 5, 6,
             7, 16, 9, 1, 2, 3, =>  9, 1, 2,
             4, 5, 6, 7, 8, 9,  =>  6, 7, 8,
             1, 2, 3, 4, 5, 6   =>  3, 4, 5,
             */
            //---
        when:
            b = a[[[0..3]:2, [1..4]:2]]
        then:
            b.toString().contains( "12.0, 4.0, 5.0, 7.0" )
            b.spread() != null
            /**
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
            p.toString().contains("5.0, 2.0, 7.0, 34.0")

        //---
        when:
            a[[[0..3]:2, [1..4]:2]] = new Tsr([2, 2], [1, 2, 3, 4])
        then:
            b.toString().contains("1.0, 2.0, 3.0, 4.0")
            a.toString().contains(
                    "1.0, 1.0, 3.0, 2.0, 5.0, 6.0, " +
                    "7.0, 16.0, 9.0, 1.0, 2.0, 3.0, " +
                    "4.0, 3.0, 6.0, 4.0, 8.0, 9.0, " +
                    "1.0, 2.0, 3.0, 4.0, 5.0, 6.0"
            )
            /**a:>>
              1   1   3   2   5   6
              7   16  9   1   2   3
              4   3   6   4   8   9
              1   2   3   4   5   6
             */
            //---
        when:
            a[1..2, 1..2] = new Tsr([2, 2], [8, 8, 8, 8])
        then:
            b.toString().contains(
                    "1.0, 2.0, "+
                    "8.0, 4.0"
            )
            a.toString().contains(
                    "1.0, 1.0, 3.0, 2.0, 5.0, 6.0, " +
                    "7.0, 8.0, 8.0, 1.0, 2.0, 3.0, " +
                    "4.0, 8.0, 8.0, 4.0, 8.0, 9.0, " +
                    "1.0, 2.0, 3.0, 4.0, 5.0, 6.0"
            )
            /**a:>>
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
            x.toString().contains(
                    "[1x1]:(20.0); ->d[2x2]:(-2.0, 3.0, 1.0, 2.0)"
            )

        where:
            device << [ Device.find('gpu'), HostCPU.instance() ]
    }




}
