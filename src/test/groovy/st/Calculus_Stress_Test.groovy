package st

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import spock.lang.Specification

class Calculus_Stress_Test extends Specification
{
    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'Stress test runs error free and produces expected result'(
        Device device, boolean arrayIndexing
    ) {
        given:
            Neureka.get().settings().indexing().isUsingArrayBasedIndexing = arrayIndexing

        and :
            def stress = ( Tsr t ) -> {
                t = t + Tsr.of( t.shape(), -3..12 )
                t = t * Tsr.of( t.shape(),  2..3 )
                t = t / Tsr.of( t.shape(),  1..2 )
                t = t ^ Tsr.of( t.shape(),  2..1 )
                t = t - Tsr.of( t.shape(), -2..2 )
                return t
            }
        and :
            Tsr t = Tsr.of( [3, 3, 3, 3], 0 ).to( device )

        when :
            t[1..2, 0..2, 1..1, 0..2] = Tsr.of( [2, 3, 1, 3], -4..2 )
            t = t[1..2, 0..2, 1..1, 0..2]

        then :
            t.toString() == Tsr.of( [2, 3, 1, 3], -4..2 ).toString()

        when :
            t = stress(t)

        then :
            t.toString().replace("E0","") == "(2x3x1x3):[" +
                        "198.0, -6.5, " +
                        "36.0, -2.5, " +
                        "2.0, 6.5, " +
                        "" +
                        "101.0, 0.0, " +
                        "15.0, 4.0, " +
                        "146.0, 13.0, " +
                        "" +
                        "400.0, 17.0, " +
                        "194.0, 15.5, " +
                        "101.0, -4.5" +
                    "]"

        where :
           device             | arrayIndexing
           CPU.get()          | true
           CPU.get()          | false
           Device.find('gpu') | true
    }


    def 'Dot operation stress test runs error free and produces expected result'(
            boolean arrayIndexing, List<Integer> shape, String expected
    ) {
        given:
            Neureka.get().settings().indexing().isUsingArrayBasedIndexing = arrayIndexing

        and :
            Tsr t = Tsr.of( shape, -4..2 )

        when :
            t = t.convDot( t.T() )

        then :
            t.toString() == expected

        where :
        arrayIndexing | shape        || expected
        true          | [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
        false         | [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
        true          | [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
        false         | [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
    }

    def 'The broadcast operation stress test runs error free and produces expected result'(
            Device device,
            boolean arrayIndexing,
            List<Integer> shape1, List<Integer> shape2,
            String operation,
            String expected
    ) {
        given:
            Neureka.get().settings().indexing().isUsingArrayBasedIndexing = arrayIndexing

        and :
            Tsr t1 = Tsr.of( shape1, -4..2 ).to( device )
            Tsr t2 = Tsr.of( shape2, -3..5 ).to( device )

        when :
            Tsr t = Tsr.of( operation, [t1,t2] )

        then :
            t.toString() == expected

        where :
            device             | arrayIndexing | shape1    | shape2    | operation || expected
            CPU.get()          | true  | [2, 1]    | [2, 2]    | 'i0%i1' || "(2x2):[-1.0, -0.0, -0.0, NaN]"
            CPU.get()          | true  | [2, 3, 1] | [1, 3, 2] | 'i0%i1' || "(2x3x2):[-1.0, -0.0, -0.0, NaN, -0.0, -0.0, -1.0, -1.0, 0.0, NaN, 0.0, 1.0]"
            CPU.get()          | false | [2, 1]    | [2, 2]    | 'i0%i1' || "(2x2):[-1.0, -0.0, -0.0, NaN]"
            CPU.get()          | false | [2, 3, 1] | [1, 3, 2] | 'i0%i1' || "(2x3x2):[-1.0, -0.0, -0.0, NaN, -0.0, -0.0, -1.0, -1.0, 0.0, NaN, 0.0, 1.0]"

            //Device.find('gpu') | true          | [2, 1]    | [2, 2]    | 'i0%i1'   || "(2x2):[-1.0, -0.0, -0.0, NaN]"
            //Device.find('gpu') | true          | [2, 3, 1] | [1, 3, 2] | 'i0%i1'   || "(2x3x2):[-1.0, -0.0, -0.0, NaN, -0.0, -0.0, -1.0, -1.0, 0.0, NaN, 0.0, 1.0]"
            //Device.find('gpu') | false         | [2, 1]    | [2, 2]    | 'i0%i1'   || "(2x2):[-1.0, -0.0, -0.0, NaN]"
            //Device.find('gpu') | false         | [2, 3, 1] | [1, 3, 2] | 'i0%i1'   || "(2x3x2):[-1.0, -0.0, -0.0, NaN, -0.0, -0.0, -1.0, -1.0, 0.0, NaN, 0.0, 1.0]"

            CPU.get()          | true  | [2, 1]    | [2, 2]    | 'i0*i1' || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            CPU.get()          | true  | [2, 3, 1] | [1, 3, 2] | 'i0*i1' || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"
            CPU.get()          | false | [2, 1]    | [2, 2]    | 'i0*i1' || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            CPU.get()          | false | [2, 3, 1] | [1, 3, 2] | 'i0*i1' || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"

            Device.find('gpu') | true          | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            Device.find('gpu') | true          | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"
            Device.find('gpu') | false         | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            Device.find('gpu') | false         | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"

            CPU.get()          | true  | [2, 1]    | [2, 2]    | 'i0+i1' || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            CPU.get()          | true  | [2, 3, 1] | [1, 3, 2] | 'i0+i1' || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"
            CPU.get()          | false | [2, 1]    | [2, 2]    | 'i0+i1' || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            CPU.get()          | false | [2, 3, 1] | [1, 3, 2] | 'i0+i1' || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"

            Device.find('gpu') | true          | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            Device.find('gpu') | true          | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"
            Device.find('gpu') | false         | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            Device.find('gpu') | false         | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"

            CPU.get()          | true  | [2, 1]    | [2, 2]    | 'i0-i1' || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            CPU.get()          | true  | [2, 3, 1] | [1, 3, 2] | 'i0-i1' || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"
            CPU.get()          | false | [2, 1]    | [2, 2]    | 'i0-i1' || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            CPU.get()          | false | [2, 3, 1] | [1, 3, 2] | 'i0-i1' || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"

            Device.find('gpu') | true          | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            Device.find('gpu') | true          | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"
            Device.find('gpu') | false         | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            Device.find('gpu') | false         | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"

            CPU.get()          | true  | [2, 1]    | [2, 2]    | 'i0/i1' || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            CPU.get()          | true  | [2, 3, 1] | [1, 3, 2] | 'i0/i1' || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
            CPU.get()          | false | [2, 1]    | [2, 2]    | 'i0/i1' || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            CPU.get()          | false | [2, 3, 1] | [1, 3, 2] | 'i0/i1' || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"

            //WIP: fix derivative! -> Make multiple kernels!
            //Device.find('gpu') | true          | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            //Device.find('gpu') | true          | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
            //Device.find('gpu') | false         | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            //Device.find('gpu') | false         | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
    }


}
