package it.calculus

import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.args.Arg
import spock.lang.Specification

class ConCat_Spec extends Specification
{
    def 'We can concatenate 2 tensors alongside a specified axis!'()
    {
        given :
            var a = Tsr.of(Double, [3, 4, 2], [-1.7, 2, 9.3, -3]).setRqsGradient(true)
            var b = Tsr.ofDoubles().withShape(3, 2, 2).andFill(3,2.5,-6)
        and :
            var cat = Function.of('concat(I[0], I[1])')

        when :
            var c = cat.callWith(Arg.Dim.of(1)).call(a, b)

        then :
            c.shape() == [3, 6, 2]

        when :
            var y = c * 2
        and :
            y.backward(-3)

        then :
            a.gradient.every( it -> it == -6 )
    }


    def 'We can concatenate 2 string tensors alongside a specified axis!'()
    {
        given :
            var a = Tsr.of(String, [2, 5], [':)', ':P', 'B)'])
            var b = Tsr.of(String).withShape(1, 5).andFill('O.o', '._.')
        and :
            var cat = Function.of('concat(I[0], I[1])')

        when :
            var c = cat.callWith(Arg.Dim.of(0)).call(a, b)

        then :
            c.shape() == [3, 5]
        and :
            c.any( it -> it == ':P' )
            c.any( it -> it == '._.' )
    }

}
