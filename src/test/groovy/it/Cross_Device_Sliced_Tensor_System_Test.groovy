package it


import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import neureka.view.NDPrintSettings
import spock.lang.Specification
import spock.lang.Title
import testutility.mock.DummyDevice

@Title("Cross Device Tensor Slicing")
class Cross_Device_Sliced_Tensor_System_Test extends Specification
{
    def setupSpec() {
        reportHeader """ 
                <p>
                    This specification covers the behavior of tensors when being sliced
                    on multiple different device types in conjunction with 
                    the autograd system.
                    Autograd should work on slices as well.          
                </p>
            """
    }


    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def 'Slices can be created using the SliceBuilder.'(
        Device device
    ) {
        given :
            if ( device == null ) return
            Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed = false
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(false)
            if ( device instanceof OpenCLDevice && !Neureka.get().canAccessOpenCLDevice() ) return

        and: 'A tensor which ought to be sliced:'
            var a = Tsr.of([4, 6], [
                            1d, 2d, 3d, 4d, 5d, 6d,
                            7d, 8d, 9d, 1d, 2d, 3d,
                            4d, 5d, 6d, 7d, 8d, 9d,
                            1d, 2d, 3d, 4d, 5d, 6d
                        ])
            /*
                Let's do the following slice! :

                    1, 2, 3, 4, 5, 6,
                    7, 8, 9, 1, 2, 3, => 7, 8, 9, 1
                    4, 5, 6, 7, 8, 9, => 4, 5, 6, 7
                    1, 2, 3, 4, 5, 6  => 1, 2, 3, 4
             */
            device.store(a)

        when:
            var b = a.slice() // [-1..-3, -6..-3]
                        .axis(0).from(-1).to(-3)
                        .axis(1).from(-6).to(-3)
                        .get()

            var s = a.slice() // [1, -2]
                        .axis(0).at(1)
                        .axis(1).at(-2)
                        .get()

            s.rqsGradient = true

        then:
            s.toString() == "(1x1):[2.0]:g:[null]"
            s.item(0) == 2.0
            s.rqsGradient()
            b.toString().contains("7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0")
            b.spread() != null

        when :
            var y = ( s * 4 ) ** 1.5

        then :
             y.toString() == '(1x1):[22.6274]; ->d(1x1):[16.9706]'

        where:
            device << [Device.get('gpu'), CPU.get() ]

    }

    def 'Cross device sliced tensor integration test runs without errors.'(
            Device device
    ) {
        given :
            if ( device == null ) return
            Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed = false
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
            if ( device instanceof OpenCLDevice && !Neureka.get().canAccessOpenCLDevice() ) return

        when :
            var x = Tsr.of([1],  3d).setRqsGradient(true)
            var b = Tsr.of([1], -4d)
            var w = Tsr.of([1],  2d)
            device.store(x).store(b).store(w)
            /*
                        ((3-4)*2)**2 = 4
                  dx:    8*3 - 32   = -8
             */
            var y = Tsr.of("((i0+i1)*i2)**2", [x, b, w])
        then:
            y.indicesMap() != null
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when:
            y.backward(Tsr.of(2d))
            y = ( ( x + b ) * w )**2

        then:
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when:
            y.backward(Tsr.of(2d))
            x.toString().contains("-32.0")
            y = b + w * x

            var a = Tsr.of([4, 6], [
                                1d, 2d, 3d, 4d, 5d, 6d,
                                7d, 8d, 9d, 1d, 2d, 3d,
                                4d, 5d, 6d, 7d, 8d, 9d,
                                1d, 2d, 3d, 4d, 5d, 6d
                        ])
            /*
                Let's do the following slice! :

                    1, 2, 3, 4, 5, 6,
                    7, 8, 9, 1, 2, 3, => 7, 8, 9, 1
                    4, 5, 6, 7, 8, 9, => 4, 5, 6, 7
                    1, 2, 3, 4, 5, 6  => 1, 2, 3, 4
             */

            device.store(a)
            b = a[[-1..-3, -6..-3]]
            var s = a[[1, -2]]

        then:
            s.toString() == "[1x1]:(2.0)"
            s.item(0) == 2.0
            b.toString().contains("7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0")
            b.spread() != null

        when:
            b = a[-3..-1, 0..3]
            s = a[1, -2]

        then:
            s.toString() == "[1x1]:(2.0)"
            s.item(0) == 2.0
            s.getDataAt(0) == 1.0
            s.getDataAt(1) == 2.0
            b.toString().contains("7.0, 8.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0")
            b.spread() != null
            /*
                As matrix :

                2, 3, 4,
                6, 7, 8,
                1, 2, 3,
                5, 6, 7,
             */

        when:
            if( device instanceof DummyDevice ) {
                a.getDataAs( double[].class )[1] = a.getDataAs( double[].class )[1] * 6
                a.getDataAs( double[].class )[7] = a.getDataAs( double[].class )[7] * 2
            } else {
                var k = Tsr.of([4, 6], [
                                1d, 6d, 1d, 1d,
                                1d, 1d, 1d, 2d,
                                1d, 1d, 1d, 1d,
                                1d, 1d, 1d, 1d,
                                1d, 1d, 1d, 1d,
                                1d, 1d, 1d, 1d
                            ])
                device.store( k )
                a.mut[] = a * k
            }

        then:
            b.toString().contains("7.0, 16.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0")

        when:
            var c = Tsr.of([3, 4], [
                            -3d, 2d, 3d,
                             5d, 6d, 2d,
                            -1d, 1d, 2d,
                             3d, 4d, 2d,
                        ])
            /*
                -3, 2,  3, 5,
                 6, 2, -1, 1,
                 2, 3,  4, 2,
                      +
                 7, 18, 9, 1
                 4, 5,  6, 7
                 1, 2,  3, 4
                      =
                 4, 20, 12, 6
                 10, 7, 5,  8
                 3,  5, 7   6

             */

            var d = b + c

        then:
            (d.NDConf.asInlineArray() as List) == ( [3, 4, 4, 1, 4, 1, 0, 0, 1, 1] )
            (b.NDConf.asInlineArray() as List) == ( [3, 4, 6, 1, 4, 1, 1, 0, 1, 1] )
            (c.NDConf.asInlineArray() as List) == ( [3, 4, 4, 1, 4, 1, 0, 0, 1, 1] )

            d.toString().contains(
                "4.0, 18.0, 12.0, 6.0, "+
                "10.0, 7.0, 5.0, 8.0, "+
                "3.0, 5.0, 7.0, 6.0"
            )

        when:
            b = a[1..3, 2..4]
        then:
            b.toString().contains("9.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0")
            b.spread() != null
            /*
                Let's do the following slice! :

                 1, 12, 3, 4, 5, 6,
                 7, 16, 9, 1, 2, 3, =>  9, 1, 2,
                 4, 5, 6, 7, 8, 9,  =>  6, 7, 8,
                 1, 2, 3, 4, 5, 6   =>  3, 4, 5,
             */

        when:
            b = a[[[0..3]:2, [1..4]:2]]
        then:
            b.toString().contains( "12.0, 4.0, 5.0, 7.0" )
            b.spread() != null
            /*
                Let's do the following slice! :

                 1, 12, 3, 4, 5, 6, => 12, 4,
                 7, 16, 9, 1, 2, 3,
                 4, 5, 6, 7, 8, 9,  => 5,  7,
                 1, 2, 3, 4, 5, 6
             */

        when:
            var p = Tsr.of([2,2], [2d, 55d, 4d, 7d]).to((device instanceof DummyDevice)?null:device)
            var u = Tsr.of([2,2], [5d, 2d, 7d, 34d]).to((device instanceof DummyDevice)?null:device)

            p.mut[] = u

        then:
            p.toString().contains("5.0, 2.0, 7.0, 34.0")

        when:
            a.mut[[[0..3]:2, [1..4]:2]] = Tsr.of([2, 2], [1d, 2d, 3d, 4d])
        then:
            b.toString().contains("1.0, 2.0, 3.0, 4.0")
            a.toString().contains(
                    "1.0, 1.0, 3.0, 2.0, 5.0, 6.0, " +
                    "7.0, 16.0, 9.0, 1.0, 2.0, 3.0, " +
                    "4.0, 3.0, 6.0, 4.0, 8.0, 9.0, " +
                    "1.0, 2.0, 3.0, 4.0, 5.0, 6.0"
            )
            /*
                a:
                  1   1   3   2   5   6
                  7   16  9   1   2   3
                  4   3   6   4   8   9
                  1   2   3   4   5   6
             */

        when:
            a.mut[1..2, 1..2] = Tsr.of([2, 2], [8, 8, 8, 8])
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
            /*
                a:
                 1.0, 1.0, 3.0, 2.0, 5.0, 6.0,
                 7.0, 8.0, 8.0, 1.0, 2.0, 3.0,
                 4.0, 8.0, 8.0, 4.0, 8.0, 9.0,
                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
                b:
                 1, 8,
                 3, 4
        */
        when:
            b.setRqsGradient(true)
            c = Tsr.of([2, 2], [
                            -2, 3,//-2 + 24 + 3 + 8
                             1, 2,
                        ])
            device.store(b).store(c) // -2 + 6 + 8 + 8 = 22
            x = Tsr.of(b, "x", c) // This test is important because it tests convolution on slices!
        then:
            x.item() == 20
        and :
            x.toString().replace(".0", "").contains("->d[2x2]:(-2, 3, 1, 2)")

        where:
            device << [CPU.get(),Device.get('gpu')]
    }



}
