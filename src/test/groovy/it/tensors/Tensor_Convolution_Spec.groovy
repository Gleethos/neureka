package it.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.common.utility.SettingsLoader
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.ndim.config.NDConfiguration
import neureka.view.TsrStringSettings
import spock.lang.IgnoreIf
import spock.lang.Specification

class Tensor_Convolution_Spec extends Specification
{
    def setup() {
        // The following is similar to Neureka.get().reset() however it uses a groovy script for library settings:
        SettingsLoader.tryGroovyScriptsOn(Neureka.get(), script -> new GroovyShell(getClass().getClassLoader()).evaluate(script))

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


    def 'The "x" (convolution) operator produces expected results (On the CPU).'(
            Class<?> type, String expected
    ) {
        reportInfo """
            The 'x' operator performs convolution on the provided operands.
            The meaning of the operands is not defined, so one the kernel tensor
            can be the first and second operand. 
        """

        given: 'Gradient auto apply for tensors in ue is set to false.'
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
        and: 'Tensor legacy view is set to true.'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
        and: 'Two new 3D tensor instances with the shapes: [2x3x1] & [1x3x2].'
            var x = Tsr.of(new int[]{2, 3, 1},
                                    new double[]{
                                        3,  2, -1,
                                        -2,  2,  4
                                    }
                                )
                                .unsafe.toType(type)

            var y = Tsr.of(new int[]{1, 3, 2},
                    new double[]{
                        4, -1,
                        3,  2,
                        3, -1
                    }
                )
                .unsafe.toType(type)

        when : 'The x-mul result is being instantiated by passing a simple equation to the tensor constructor.'
            var z = Tsr.of("I0xi1", x, y)
        then: 'The result contains the expected String.'
            z.toString().contains(expected)

        when: 'The x-mul result is being instantiated by passing a object array containing equation parameters and syntax.'
            z = Tsr.of(new Object[]{x, "x", y})
        then: 'The result contains the expected String.'
            z.toString().contains(expected)

        where :
            type   || expected
            Double || "[2x1x2]:(15.0, 2.0, 10.0, 2.0)"
            Float  || "[2x1x2]:(15.0, 2.0, 10.0, 2.0)"
    }


    def 'Manual convolution produces expected result.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
            Tsr a = Tsr.of([100, 100], 3d..19d)
            Tsr x = a[1..-2,0..-1]
            Tsr y = a[0..-3,0..-1]
            Tsr z = a[2..-1,0..-1]

        when :
            Tsr rowconvol = x + y + z // (98, 100) (98, 100) (98, 100)
            Tsr k = rowconvol[0..-1,1..-2]
            Tsr v = rowconvol[0..-1,0..-3]
            Tsr j = rowconvol[0..-1,2..-1]
            Tsr u = a[1..-2,1..-2]
            Tsr colconvol = k + v + j - 9 * u // (98, 98)+(98, 98)+(98, 98)-9*(98, 98)
            String xAsStr = x.toString()
            String yAsStr = y.toString()
            String zAsStr = z.toString()
            String rcAsStr = rowconvol.toString()
            String kAsStr = k.toString()
            String vAsStr = v.toString()
            String jAsStr = j.toString()
            String uAsStr = u.toString()

        then :
            xAsStr.contains("(98x100):[18.0, 19.0, 3.0, 4.0, 5.0")
            yAsStr.contains("(98x100):[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0")
            zAsStr.contains("(98x100):[16.0, 17.0, 18.0, 19.0, 3.0")
            rcAsStr.contains("(98x100):[37.0, 40.0, 26.0, 29.0, 15.0, 18.0")
            kAsStr.contains("(98x98):[40.0, 26.0, 29.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0")
            vAsStr.contains("(98x98):[37.0, 40.0, 26.0, 29.0, 15.0, 18.0, 21.0")
            jAsStr.contains("(98x98):[26.0, 29.0, 15.0, 18.0, 21.0, 24.0")
            uAsStr.contains("(98x98):[19.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, ")
            colconvol.toString().contains("(98x98):[-68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, " +
                    "-34.0, -68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, -34.0, " +
                    "-68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, ... + 9554 more]")
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null }) // We need to assure that this system supports OpenCL!
    def 'Very simple manual convolution produces expected result.'(
            Device device
    ) {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
            Tsr a = Tsr.of([4, 4], 0d..16d).to( device )

            Tsr x = a[1..-2,0..-1]
            Tsr y = a[0..-3,0..-1]
            Tsr z = a[2..-1,0..-1]

        when :
            Tsr rowconvol = x + y + z
            Tsr k = rowconvol[0..-1,1..-2]
            Tsr v = rowconvol[0..-1,0..-3]
            Tsr j = rowconvol[0..-1,2..-1]
            Tsr u = a[1..-2,1..-2]
            Tsr colconvol = k + v + j - 9 * u
            String xAsStr = x.toString()
            String yAsStr = y.toString()
            String zAsStr = z.toString()
            String rcAsStr = rowconvol.toString()
            String kAsStr = k.toString()
            String vAsStr = v.toString()
            String jAsStr = j.toString()
            String uAsStr = u.toString()

        then :
            xAsStr.contains("(2x4):[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]")
            yAsStr.contains("(2x4):[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]")
            zAsStr.contains("(2x4):[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]")
            rcAsStr.contains("(2x4):[12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0]")
            kAsStr.contains("(2x2):[15.0, 18.0, 27.0, 30.0]")
            vAsStr.contains("(2x2):[12.0, 15.0, 24.0, 27.0]")
            jAsStr.contains("(2x2):[18.0, 21.0, 30.0, 33.0]")
            uAsStr.contains("(2x2):[5.0, 6.0, 9.0, 10.0]")
            colconvol.toString().contains("(2x2):[0.0, 0.0, 0.0, 0.0]")

        where : 'The following data is being used for tensor instantiation :'
            device  << [CPU.get(), Device.get("openCL") ]
    }


    void 'Autograd works with simple 2D convolution.'()
    {
        given : 'The following library configuration is being used.'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(false)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(true)

        when : 'The following calculations are being executed...'
            Tsr<Double> i_a = Tsr.of([2, 1], [
                                                1d,
                                                2d
                                            ])
            Tsr<Double> w_a = Tsr.of([2, 2], [
                                    1d, 3d,
                                    4d, -1d
                            ]).setRqsGradient(true)
            Tsr<Double> o_a = Tsr.of(i_a, "x", w_a)
            //[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0),
            //---
            Tsr<Double> w_b = Tsr.of([2, 2], [
                                    -2d, 1d,  // 9, 1 -> -17
                                    2d, -1d   // ... -> 17
                            ]).setRqsGradient(true)
            Tsr o_b = Tsr.of(o_a, "x", w_b)
            //[2x1]:(-10.0, 5.0); ->d[2x2]:(-2.0, 1.0, 2.0, -1.0):g:(null), ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), ,
            //---
            Tsr w_c = Tsr.of([2, 2], [
                                    0.5d, 3d,
                                    -2d, -0.5d
                            ]).setRqsGradient(true)
            Tsr o_c = Tsr.of(o_a, "x", w_c)
            //[2x1]:(-0.5, 20.0); ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), , ->d[2x2]:(0.5, 3.0, -2.0, -0.5):g:(null),
            //---
            Tsr out = o_b * o_c

        then : 'The results are as expected.'
            o_a.toString().contains("(9.0, 1.0)")
            out.toString().contains("(-127.5, -314.5)")
            o_b.toString().contains("(-17.0, 17.0)")
            o_c.toString().contains("(7.5, -18.5)")

            w_a.toString().contains("g:(null)")
            w_b.toString().contains("g:(null)")

        when : 'The "backward" method is being called on the "out" tensor...'
            out.backward(Tsr.of([2, 1], 1d))

        then : 'The autograd system produces the expected results.'
            w_a.toString().contains("g:(null)")
            !w_b.toString().contains("g:(null)")

        when : 'Neureka is being configured to apply tensors when host tensor is being used...'
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            w_a * 3
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)

        then : 'The tensors change their states as expected.'
            w_a.toString().contains("g:(null)")
            w_a.toString().contains("(-93.5, -30.5, -185.0, -68.0):g:(null)")
            !w_b.toString().contains("g:(null)")
            w_b.toString().contains("g:(67.5, 7.5, -166.5, -18.5)")
        //TODO: calculate size and errors and check correctness!
    }


    def 'Sime convolution works as expected eith autograd.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(false)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(true)

        when :
            Tsr i_a = Tsr.of([2, 1], [1d, 2d])
            Tsr w_a = Tsr.of([2, 2], [1d, 3d, 4d, -1d]).setRqsGradient(true)
            Tsr o_a = Tsr.of(i_a,"x", w_a)
            //[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0),
            //---
            Tsr w_b = Tsr.of([2, 2], [-2d, 1d, 2d, -1d]).setRqsGradient(true)
            Tsr o_b = Tsr.of(o_a,"x", w_b)
            //[2x1]:(-10.0, 5.0); ->d[2x2]:(-2.0, 1.0, 2.0, -1.0):g:(null), ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), ,
            //---
            Tsr w_c = Tsr.of([2, 2], [0.5d, 3d, -2d, -0.5d]).setRqsGradient(true)
            Tsr o_c = Tsr.of(o_a, "x", w_c)
            //[2x1]:(-0.5, 20.0); ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), , ->d[2x2]:(0.5, 3.0, -2.0, -0.5):g:(null),
            //---
            Tsr out = o_b*o_c

        then :
            o_a.toString().contains("(9.0, 1.0)")
            out.toString().contains("(-127.5, -314.5)")
            o_b.toString().contains("(-17.0, 17.0)")
            o_c.toString().contains("(7.5, -18.5)")

            w_a.toString().contains("g:(null)")
            w_b.toString().contains("g:(null)")

        when :
            out.backward(Tsr.of([2, 1], 1))

        then :
            w_a.toString().contains("g:(null)")
            !w_b.toString().contains("g:(null)")

        when :
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            w_a * 3
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)

        then :
            w_a.toString().contains("g:(null)")
            !w_a.toString().contains("1.0, 3.0, 4.0, -1.0")
            !w_b.toString().contains("g:(null)")
            //TODO: calculate size and errors and check correctness!
    }


    def 'Tensors have the correct layout after convolution.'()
    {
        given :
            Tsr<Double> t0 = Tsr.of([3, 2, 1], [
                                        1d, 2d,
                                        3d, 4d,
                                        5d, 6d
                                ])
            Tsr<Double> x0 = Tsr.of([1, 2, 3], [
                                        1d, 2d, 3d,
                                        4d, 5d, 6d
                                ])
            /*
                    9   12  15
                    19  26  33
                    29  40  51
             */

        expect :
            t0.unsafe.data == [1, 2, 3, 4, 5, 6] as double[]
            x0.unsafe.data == [1, 2, 3, 4, 5, 6] as double[]
            t0.NDConf.layout == NDConfiguration.Layout.ROW_MAJOR
            x0.NDConf.layout == NDConfiguration.Layout.ROW_MAJOR

        when : Tsr<Double> out0 = Tsr.of("i0xi1", [t0, x0] )
        then :
            out0.toString() == "(3x1x3):[9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0]"

        when :
            t0.unsafe.toLayout(NDConfiguration.Layout.COLUMN_MAJOR)
            x0.unsafe.toLayout(NDConfiguration.Layout.COLUMN_MAJOR)
        then :
            t0.NDConf.layout == NDConfiguration.Layout.COLUMN_MAJOR
            x0.NDConf.layout == NDConfiguration.Layout.COLUMN_MAJOR
            t0.unsafe.data == [1, 3, 5, 2, 4, 6] as double[]
            x0.unsafe.data == [1, 4, 2, 5, 3, 6] as double[]
        and :
            t0.toString() == "(3x2x1):[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]"
            x0.toString() == "(1x2x3):[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]"

        when : out0 = Tsr.of("i0xi1", [t0, x0])
        then :
            out0.toString() == "(3x1x3):[9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0]"
    }

}
