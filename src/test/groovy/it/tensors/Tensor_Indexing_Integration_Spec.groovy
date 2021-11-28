package it.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.view.TsrStringSettings
import spock.lang.Specification

class Tensor_Indexing_Integration_Spec extends Specification
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

    void 'Test convolution with legacy indexing.'()
    {
        given : 'The following library configuration is being used.'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(false)

        when : 'The following calculations are being executed...'
            Tsr i_a = Tsr.of([2, 1], [
                    1,
                    2
            ])
            Tsr w_a = Tsr.of([2, 2], [
                    1, 3,
                    4, -1
            ]).setRqsGradient(true)
            Tsr o_a = Tsr.of(i_a, "x", w_a)
            //[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0),
            //---
            Tsr w_b = Tsr.of([2, 2], [
                    -2, 1,  // 9, 1 -> -17
                    2, -1   // ... -> 17
            ]).setRqsGradient(true)
            Tsr o_b = Tsr.of(o_a, "x", w_b)
            //[2x1]:(-10.0, 5.0); ->d[2x2]:(-2.0, 1.0, 2.0, -1.0):g:(null), ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), ,
            //---
            Tsr w_c = Tsr.of([2, 2], [
                    0.5, 3,
                    -2, -0.5
            ]).setRqsGradient(true)
            Tsr o_c = Tsr.of(o_a, "x", w_c)
            //[2x1]:(-0.5, 20.0); ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), , ->d[2x2]:(0.5, 3.0, -2.0, -0.5):g:(null),
            //---
            Tsr out = o_b * o_c

        then : 'The results are as expected.'
            assert o_a.toString().contains("(9.0, 1.0)")
            assert out.toString().contains("(-127.5, -314.5)")
            assert o_b.toString().contains("(-17.0, 17.0)")
            assert o_c.toString().contains("(7.5, -18.5)")

            assert w_a.toString().contains("g:(null)")
            assert w_b.toString().contains("g:(null)")

        when : 'The "backward" method is being called on the "out" tensor...'
            out.backward(Tsr.of([2, 1], 1))

        then : 'The autograd system produces the expected results.'
            assert w_a.toString().contains("g:(null)")
            assert !w_b.toString().contains("g:(null)")

        when : 'Neureka is being configured to apply tensors when host tensor is being used...'
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            w_a * 3
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)

        then : 'The tensors change their states as expected.'
            assert w_a.toString().contains("g:(null)")
            assert w_a.toString().contains("(-93.5, -30.5, -185.0, -68.0):g:(null)")
            assert !w_b.toString().contains("g:(null)")
            assert w_b.toString().contains("g:(67.5, 7.5, -166.5, -18.5)")
        //TODO: calculate size and errors and check correctness!
    }



    def 'Convolution using legacy indexing works as expected.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(false)

        when :
            Tsr i_a = Tsr.of([2, 1], [1, 2])
            Tsr w_a = Tsr.of([2, 2], [1, 3, 4, -1]).setRqsGradient(true)
            Tsr o_a = Tsr.of(i_a,"x", w_a)
            //[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0),
            //---
            Tsr w_b = Tsr.of([2, 2], [-2, 1, 2, -1]).setRqsGradient(true)
            Tsr o_b = Tsr.of(o_a,"x", w_b)
            //[2x1]:(-10.0, 5.0); ->d[2x2]:(-2.0, 1.0, 2.0, -1.0):g:(null), ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), ,
            //---
            Tsr w_c = Tsr.of([2, 2], [0.5, 3, -2, -0.5]).setRqsGradient(true)
            Tsr o_c = Tsr.of(o_a, "x", w_c)
            //[2x1]:(-0.5, 20.0); ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), , ->d[2x2]:(0.5, 3.0, -2.0, -0.5):g:(null),
            //---
            Tsr out = o_b*o_c

        then :
            assert o_a.toString().contains("(9.0, 1.0)")
            assert out.toString().contains("(-127.5, -314.5)")
            assert o_b.toString().contains("(-17.0, 17.0)")
            assert o_c.toString().contains("(7.5, -18.5)")

            assert w_a.toString().contains("g:(null)")
            assert w_b.toString().contains("g:(null)")

        when :
            out.backward(Tsr.of([2, 1], 1))

        then :
            assert w_a.toString().contains("g:(null)")
            assert !w_b.toString().contains("g:(null)")

        when :
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            w_a * 3
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)

        then :
            assert w_a.toString().contains("g:(null)")
            assert !w_a.toString().contains("1.0, 3.0, 4.0, -1.0")
            assert !w_b.toString().contains("g:(null)")
            //TODO: calculate size and errors and check correctness!
    }


    def 'Indexing modes produce expected results when doing convolution.'()
    {
        given :
            Tsr t0 = Tsr.of([3, 2, 1], [
                    1, 2,
                    3, 4,
                    5, 6
            ])
            Tsr x0 = Tsr.of([1, 2, 3], [
                    1, 2, 3,
                    4, 5, 6
            ])
            /*
                    9   12  15
                    19  26  33
                    29  40  51
             */
        when : Tsr out0 = Tsr.of("i0xi1", [t0, x0] )
        then :
            out0.toString() == "(3x1x3):[9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0]"
    }

}
