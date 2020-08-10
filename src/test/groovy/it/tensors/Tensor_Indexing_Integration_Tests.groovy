package it.tensors

import neureka.Neureka
import neureka.Tsr
import spock.lang.Specification

class Tensor_Indexing_Integration_Tests extends Specification
{

    void 'Test convolution with legacy indexing.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true)
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenRequested(false)

        when :
            Tsr i_a = new Tsr([2, 1], [1, 2])
            Tsr w_a = new Tsr([2, 2], [1, 3, 4, -1]).setRqsGradient(true)
            Tsr o_a = new Tsr(i_a, "x", w_a)
            //[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0),
            //---
            Tsr w_b = new Tsr([2, 2], [-2, 1, 2, -1]).setRqsGradient(true)
            Tsr o_b = new Tsr(o_a, "x", w_b)
            //[2x1]:(-10.0, 5.0); ->d[2x2]:(-2.0, 1.0, 2.0, -1.0):g:(null), ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), ,
            //---
            Tsr w_c = new Tsr([2, 2], [0.5, 3, -2, -0.5]).setRqsGradient(true)
            Tsr o_c = new Tsr(o_a, "x", w_c)
            //[2x1]:(-0.5, 20.0); ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), , ->d[2x2]:(0.5, 3.0, -2.0, -0.5):g:(null),
            //---
            Tsr out = o_b * o_c

        then :
            assert o_a.toString().contains("(7.0, 2.0)")
            assert out.toString().contains("(5.0, 100.0)")
            assert o_b.toString().contains("(-10.0, 5.0)")
            assert o_c.toString().contains("(-0.5, 20.0)")

            assert w_a.toString().contains("g:(null)")
            assert w_b.toString().contains("g:(null)")

        when : out.backward(new Tsr([2, 1], 1))

        then :
            assert w_a.toString().contains("g:(null)")
            assert !w_b.toString().contains("g:(null)")

        when :
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            w_a * 3
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)

        then :
            assert w_a.toString().contains("g:(null)")
            assert !w_a.toString().contains("1.0, 3.0, 4.0, -1.0")
            assert !w_b.toString().contains("g:(null)")
        //TODO: calculate size and errors and check correctness!
    }



    def 'Convolution using legacy indexing works as expected.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false)
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenRequested(false)

        when :
            Tsr i_a = new Tsr([2, 1], [1, 2])
            Tsr w_a = new Tsr([2, 2], [1, 3, 4, -1]).setRqsGradient(true)
            Tsr o_a = new Tsr(i_a,"x", w_a)
            //[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0),
            //---
            Tsr w_b = new Tsr([2, 2], [-2, 1, 2, -1]).setRqsGradient(true)
            Tsr o_b = new Tsr(o_a,"x", w_b)
            //[2x1]:(-10.0, 5.0); ->d[2x2]:(-2.0, 1.0, 2.0, -1.0):g:(null), ->d[1x2]:(7.0, 2.0); ->d[2x1]:(1.0, 2.0), ,
            //---
            Tsr w_c = new Tsr([2, 2], [0.5, 3, -2, -0.5]).setRqsGradient(true)
            Tsr o_c = new Tsr(o_a, "x", w_c)
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
            out.backward(new Tsr([2, 1], 1))

        then :
            assert w_a.toString().contains("g:(null)")
            assert !w_b.toString().contains("g:(null)")

        when :
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            w_a * 3
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)

        then :
            assert w_a.toString().contains("g:(null)")
            assert !w_a.toString().contains("1.0, 3.0, 4.0, -1.0")
            assert !w_b.toString().contains("g:(null)")
            //TODO: calculate size and errors and check correctness!
    }


    def 'Indexing modes produce expected results when doing convolution.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false)
            Tsr t0 = new Tsr([3, 2, 1], [
                    1, 2,
                    3, 4,
                    5, 6
            ])
            Tsr x0 = new Tsr([1, 2, 3], [
                    1, 2, 3,
                    4, 5, 6
            ])
            /*
                    9   12  15
                    19  26  33
                    29  40  51
             */
        when : Tsr out0 = new Tsr([t0, x0], "i0xi1")
        then : out0.toString().equals("(3x1x3):[9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0]")
        and :
            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true)
            Tsr t1 = new Tsr([3, 2, 1], [
                    1, 2, 3,
                    4, 5, 6
            ])
            Tsr x1 = new Tsr([1, 2, 3], [
                    1, 2,
                    3, 4,
                    5, 6
            ])
            /*
                    9   12  15
                    19  26  33
                    26  40  51
             */
        when : Tsr out1 = new Tsr([t0, x0], "i0xi1")
        then : out1.toString().equals("(3x1x3):[9.0, 19.0, 29.0, 12.0, 26.0, 40.0, 15.0, 33.0, 51.0]")
    }

}
