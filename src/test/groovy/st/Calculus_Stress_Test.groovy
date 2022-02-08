package st

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import neureka.view.TsrStringSettings
import spock.lang.Specification

class Calculus_Stress_Test extends Specification
{
    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
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

    def 'Stress test runs error free and produces expected result'(
        Device device
    ) {
        given:
            def stress = ( Tsr t ) -> {
                t = t + Tsr.of( t.shape(), -3..12 )
                t = t * Tsr.of( t.shape(),  2..3  )
                t = t / Tsr.of( t.shape(),  1..2  )
                t = t ^ Tsr.of( t.shape(),  2..1  )
                t = t - Tsr.of( t.shape(), -2..2  )
                return t
            }
        and :
            Tsr source = Tsr.of( [3, 3, 3, 3], -1 ).to( device )

        when :
            source[1..2, 0..2, 1..1, 0..2] = Tsr.of( [2, 3, 1, 3], -4..2 )
            Tsr t = source[1..2, 0..2, 1..1, 0..2]

        then :
            t.toString() == Tsr.of( [2, 3, 1, 3], -4..2 ).toString()

        when :
            t = stress(t)

        then :
            t.toString({it.hasSlimNumbers = true}) ==
                    "(2x3x1x3):[" +
                        "198, -6.5, " +
                        "36, -2.5, " +
                        "2, 6.5, " +
                        "" +
                        "101, 0, " +
                        "15, 4, " +
                        "146, 13, " +
                        "" +
                        "400, 17, " +
                        "194, 15.5, " +
                        "101, -4.5" +
                    "]"
        and :
            (device instanceof OpenCLDevice) || t.data == [198.0, -6.5, 36.0, -2.5, 2.0, 6.5, 101.0, 0.0, 15.0, 4.0, 146.0, 13.0, 400.0, 17.0, 194.0, 15.5, 101.0, -4.5]
            (device instanceof OpenCLDevice) || source.data == [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -4.0, -3.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, -4.0, -3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -3.0, -2.0, -1.0, -1.0, -1.0, -1.0]

        where :
           device << [CPU.get(), Device.find('gpu')]
    }


    def 'Dot operation stress test runs error free and produces expected result'(
            List<Integer> shape, String expected
    ) {
        given:
            Tsr<Double> t = Tsr.of( shape, -4d..2d )

        when :
            t = t.convDot( t.T() )

        then :
            t.toString() == expected

        where :
            shape        || expected
            [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
            [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
            [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
            [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
    }

    def 'The broadcast operation stress test runs error free and produces expected result'(
            Device device,
            List<Integer> shape1, List<Integer> shape2,
            String operation,
            String expected
    ) {
        given:
            Tsr<Double> t1 = Tsr.of( shape1, -4d..2d ).to( device )
            Tsr<Double> t2 = Tsr.of( shape2, -3d..5d ).to( device )

        when :
            Tsr t = Tsr.of( operation, [t1,t2] )

        then :
            t.toString() == expected

        where :
            device             | shape1    | shape2    | operation || expected
            CPU.get()          | [2, 1]    | [2, 2]    | 'i0%i1' || "(2x2):[-1.0, -0.0, -0.0, NaN]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0%i1' || "(2x3x2):[-1.0, -0.0, -0.0, NaN, -0.0, -0.0, -1.0, -1.0, 0.0, NaN, 0.0, 1.0]"

            //Device.find('gpu') | [2, 1]    | [2, 2]    | 'i0%i1'   || "(2x2):[-1.0, -0.0, -0.0, NaN]"
            //Device.find('gpu') | [2, 3, 1] | [1, 3, 2] | 'i0%i1'   || "(2x3x2):[-1.0, -0.0, -0.0, NaN, -0.0, -0.0, -1.0, -1.0, 0.0, NaN, 0.0, 1.0]"

            CPU.get()          | [2, 1]    | [2, 2]    | 'i0*i1' || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0*i1' || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"

            Device.find('gpu') | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            Device.find('gpu') | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"

            CPU.get()          | [2, 1]    | [2, 2]    | 'i0+i1' || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0+i1' || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"

            Device.find('gpu') | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            Device.find('gpu') | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"

            CPU.get()          | [2, 1]    | [2, 2]    | 'i0-i1' || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0-i1' || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"

            Device.find('gpu') | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            Device.find('gpu') | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"

            CPU.get()          | [2, 1]    | [2, 2]    | 'i0/i1' || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0/i1' || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"

            //WIP: fix derivative! -> Make multiple kernels!
            //Device.find('gpu') | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            //Device.find('gpu') | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
    }


}
