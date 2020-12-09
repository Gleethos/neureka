package st

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.HostCPU
import neureka.devices.opencl.OpenCLDevice
import neureka.devices.opencl.OpenCLPlatform
import spock.lang.Specification

class Calculus_Stress_Test extends Specification
{


    def 'Stress test runs error free and produces expected result'(
        Device device, boolean arrayIndexing, boolean legacyIndexing
    ) {
        given: 'Neureka is being reset.'
            Neureka.instance().reset()
        and:
            Neureka.instance().settings().indexing().isUsingArrayBasedIndexing = arrayIndexing
            Neureka.instance().settings().indexing().isUsingLegacyIndexing = legacyIndexing
        and :
            if (
                device instanceof OpenCLDevice &&
                        ( (OpenCLDevice) device ).getPlatform().isDoingLegacyIndexing() != legacyIndexing
            ) ( (OpenCLDevice) device ).getPlatform().recompile()
        and :
            def stress = ( Tsr t ) -> {
                t = t + new Tsr( t.shape(), -3..12 )
                t = t * new Tsr( t.shape(),  2..3 )
                t = t / new Tsr( t.shape(),  1..2 )
                t = t ^ new Tsr( t.shape(),  2..1 )
                t = t - new Tsr( t.shape(), -2..2 )
                return t
            }
        and :
            Tsr t = new Tsr( [3, 3, 3, 3], 0 ).set( device )

        when :
            t[1..2, 0..2, 1..1, 0..2] = new Tsr( [2, 3, 1, 3], -4..2 )
            t = t[1..2, 0..2, 1..1, 0..2]

        then :
            t.toString() == new Tsr( [2, 3, 1, 3], -4..2 ).toString()

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
           device             | arrayIndexing | legacyIndexing
           HostCPU.instance() | true          | true
           HostCPU.instance() | true          | false
           HostCPU.instance() | false         | true
           HostCPU.instance() | false         | false
           Device.find('gpu') | true          | true
           Device.find('gpu') | true          | false
    }


    def 'Dot operation stress test runs error free and produces expected result'(
            boolean arrayIndexing, boolean legacyIndexing, List<Integer> shape, String expected
    ) {
        given: 'Neureka is being reset.'
            Neureka.instance().reset()
        and:
            Neureka.instance().settings().indexing().isUsingArrayBasedIndexing = arrayIndexing
            Neureka.instance().settings().indexing().isUsingLegacyIndexing = legacyIndexing

        and :
            Tsr t = new Tsr( shape, -4..2 )

        when :
            t = t.dot( t.T() )

        then :
            t.toString() == expected

        where :
        arrayIndexing | legacyIndexing | shape        || expected
        true          | true           | [2, 3]       || "(2x1x2):[20.0, 14.0, 14.0, 11.0]"
        true          | false          | [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
        false         | true           | [2, 3]       || "(2x1x2):[20.0, 14.0, 14.0, 11.0]"
        false         | false          | [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
        true          | true           | [2, 1, 3]    || "(2x1x1x1x2):[20.0, 14.0, 14.0, 11.0]"
        true          | false          | [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
        false         | true           | [2, 1, 3]    || "(2x1x1x1x2):[20.0, 14.0, 14.0, 11.0]"
        false         | false          | [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
    }

    def 'The broadcast operation stress test runs error free and produces expected result'(
            Device device,
            boolean arrayIndexing, boolean legacyIndexing,
            List<Integer> shape1, List<Integer> shape2,
            String operation,
            String expected
    ) {
        given: 'Neureka is being reset.'
            Neureka.instance().reset()
        and:
            if (
                device instanceof OpenCLDevice &&
                    ( (OpenCLDevice) device ).getPlatform().isDoingLegacyIndexing() != legacyIndexing
            ) ( (OpenCLDevice) device ).getPlatform().recompile()
            Neureka.instance().settings().indexing().isUsingArrayBasedIndexing = arrayIndexing
            Neureka.instance().settings().indexing().isUsingLegacyIndexing = legacyIndexing

        and :
            Tsr t1 = new Tsr( shape1, -4..2 ).set( device )
            Tsr t2 = new Tsr( shape2, -3..5 ).set( device )

        when :
            Tsr t = new Tsr( operation, [t1,t2] )

        then :
            t.toString() == expected

        where :
            device             | arrayIndexing | legacyIndexing | shape1    | shape2    | operation || expected
            HostCPU.instance() | true          | false          | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            HostCPU.instance() | true          | true           | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 6.0, 4.0, -0.0]"
            HostCPU.instance() | true          | false          | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"
            HostCPU.instance() | true          | true           | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 9.0, 4.0, 2.0, -0.0, -1.0, -0.0, -0.0, -2.0, -1.0, 0.0, 2.0]"
            HostCPU.instance() | false         | false          | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            HostCPU.instance() | false         | true           | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 6.0, 4.0, -0.0]"
            HostCPU.instance() | false         | false          | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"
            HostCPU.instance() | false         | true           | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 9.0, 4.0, 2.0, -0.0, -1.0, -0.0, -0.0, -2.0, -1.0, 0.0, 2.0]"

            Device.find('gpu') | true          | false          | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            Device.find('gpu') | true          | false          | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"
            Device.find('gpu') | false         | false          | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            Device.find('gpu') | false         | false          | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"
            //WIP:
            //Device.find('gpu') | true          | true           | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 6.0, 4.0, -0.0]"
            //Device.find('gpu') | true          | true           | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 9.0, 4.0, 2.0, -0.0, -1.0, -0.0, -0.0, -2.0, -1.0, 0.0, 2.0]"
            //Device.find('gpu') | false         | true           | [2, 1]    | [2, 2]    | 'i0*i1'   || "(2x2):[12.0, 6.0, 4.0, -0.0]"
            //Device.find('gpu') | false         | true           | [2, 3, 1] | [1, 3, 2] | 'i0*i1'   || "(2x3x2):[12.0, 9.0, 4.0, 2.0, -0.0, -1.0, -0.0, -0.0, -2.0, -1.0, 0.0, 2.0]"

            // TODO : check with legacy (inverse)
            HostCPU.instance() | true          | false          | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            HostCPU.instance() | true          | true           | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -5.0, -5.0, -3.0]"
            HostCPU.instance() | true          | false          | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"
            HostCPU.instance() | true          | true           | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"
            HostCPU.instance() | false         | false          | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            HostCPU.instance() | false         | true           | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -5.0, -5.0, -3.0]"
            HostCPU.instance() | false         | false          | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"
            HostCPU.instance() | false         | true           | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"

            Device.find('gpu') | true          | false          | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            Device.find('gpu') | true          | false          | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"
            Device.find('gpu') | false         | false          | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            Device.find('gpu') | false         | false          | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"
            //WIP:
            //Device.find('gpu') | true          | true           | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -5.0, -5.0, -3.0]"
            //Device.find('gpu') | true          | true           | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"
            //Device.find('gpu') | false         | true           | [2, 1]    | [2, 2]    | 'i0+i1'   || "(2x2):[-7.0, -5.0, -5.0, -3.0]"
            //Device.find('gpu') | false         | true           | [2, 3, 1] | [1, 3, 2] | 'i0+i1'   || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"

            // TODO : check with legacy (inverse)
            HostCPU.instance() | true          | false          | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            HostCPU.instance() | true          | true           | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -1.0, -3.0, -3.0]"
            HostCPU.instance() | true          | false          | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"
            HostCPU.instance() | true          | true           | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, 0.0, 0.0, 1.0, 1.0, 2.0, -4.0, -3.0, -3.0, -2.0, -2.0, -1.0]"
            HostCPU.instance() | false         | false          | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            HostCPU.instance() | false         | true           | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -1.0, -3.0, -3.0]"
            HostCPU.instance() | false         | false          | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"
            HostCPU.instance() | false         | true           | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, 0.0, 0.0, 1.0, 1.0, 2.0, -4.0, -3.0, -3.0, -2.0, -2.0, -1.0]"

            Device.find('gpu') | true          | false          | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            Device.find('gpu') | true          | false          | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"
            Device.find('gpu') | false         | false          | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            Device.find('gpu') | false         | false          | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"
            //WIP:
            //Device.find('gpu') | true          | true           | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -1.0, -3.0, -3.0]"
            //Device.find('gpu') | true          | true           | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, 0.0, 0.0, 1.0, 1.0, 2.0, -4.0, -3.0, -3.0, -2.0, -2.0, -1.0]"
            //Device.find('gpu') | false         | true           | [2, 1]    | [2, 2]    | 'i0-i1'   || "(2x2):[-1.0, -1.0, -3.0, -3.0]"
            //Device.find('gpu') | false         | true           | [2, 3, 1] | [1, 3, 2] | 'i0-i1'   || "(2x3x2):[-1.0, 0.0, 0.0, 1.0, 1.0, 2.0, -4.0, -3.0, -3.0, -2.0, -2.0, -1.0]"

            // TODO : check with legacy (inverse)
            HostCPU.instance() | true          | false          | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            HostCPU.instance() | true          | true           | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 1.5, 4.0, -∞]"
            HostCPU.instance() | true          | false          | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
            HostCPU.instance() | true          | true           | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 1.0, 1.0, 0.5, -0.0, -1.0, -∞, -∞, -2.0, -1.0, 0.0, 0.5]"
            HostCPU.instance() | false         | false          | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            HostCPU.instance() | false         | true           | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 1.5, 4.0, -∞]"
            HostCPU.instance() | false         | false          | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
            HostCPU.instance() | false         | true           | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 1.0, 1.0, 0.5, -0.0, -1.0, -∞, -∞, -2.0, -1.0, 0.0, 0.5]"

            //WIP: fix derivative! -> Make multiple kernels!
            //Device.find('gpu') | true          | false          | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            //Device.find('gpu') | true          | false          | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
            //Device.find('gpu') | false         | false          | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            //Device.find('gpu') | false         | false          | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
            //WIP:
            //Device.find('gpu') | true          | true           | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 1.5, 4.0, -∞]"
            //Device.find('gpu') | true          | true           | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 1.0, 1.0, 0.5, -0.0, -1.0, -∞, -∞, -2.0, -1.0, 0.0, 0.5]"
            //Device.find('gpu') | false         | true           | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 1.5, 4.0, -∞]"
            //Device.find('gpu') | false         | true           | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 1.0, 1.0, 0.5, -0.0, -1.0, -∞, -∞, -2.0, -1.0, 0.0, 0.5]"
    }


}
