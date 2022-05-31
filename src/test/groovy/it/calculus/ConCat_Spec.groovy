package it.calculus

import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.args.Arg
import spock.lang.Specification

class ConCat_Spec extends Specification
{
    def 'We can concatenate 2 tensors alongside a specified axis!'()
    {
        given : 'We create 2 rank 3 tensors, which we want to concatinate.'
            var a = Tsr.of(Double, [3, 4, 2], [-1.7, 2, 9.3, -3]).setRqsGradient(true)
            var b = Tsr.ofDoubles().withShape(3, 2, 2).andFill(3,2.5,-6)
        and : 'A function which should perform the concatenation.'
            var cat = Function.of('concat(I[0], I[1])')

        when : 'We call the previously created function alongside the axis alongside we want to concatinate '
            var c = cat.callWith(Arg.Axis.of(1)).call(a, b)

        then : 'The resulting tensor should have the expected shape.'
            c.shape() == [3, 6, 2]

        when : 'We use the result for some more operations...'
            var y = c * 2
        and : 'We back-propagate -3 on y...'
            y.backward(-3)

        then : 'The gradient of the furst tensor should look as follows!'
            a.gradient.every( it -> it == -6 )
    }

    def 'We can concatenate 2 float tensors alongside a specified axis!'()
    {
        given :
            var a = Tsr.of(Float, [4, 2], [8f, -9f, 5f]).setRqsGradient(true)
            var b = Tsr.ofFloats().withShape(4, 3).andFill(1,6,-6,3).setRqsGradient(true)
        and :
            var cat = Function.of('concat(I[0], I[1])')

        when :
            var c = cat.callWith(Arg.Axis.of(1)).call(a, b)

        then :
            c.shape() == [4, 5]

        when :
            var y = c * 5 + 1
        and :
            y.backward(-2)

        then :
            a.gradient.every( it -> it == -10 )
            b.gradient.every( it -> it == -10 )
    }


    def 'We can concatenate 2 string tensors alongside a specified axis!'()
    {
        given :
            var a = Tsr.of(String, [2, 5], [':)', ':P', 'B)'])
            var b = Tsr.of(String).withShape(1, 5).andFill('O.o', '._.')
        and :
            var cat = Function.of('concat(I[0], I[1])')

        when :
            var c = cat.callWith(Arg.Axis.of(0)).call(a, b)

        then :
            c.shape() == [3, 5]
        and :
            c.any( it -> it == ':P' )
            c.any( it -> it == '._.' )
    }


    def 'We can concatenate and then back-propagate 2 simple float tensors alongside a specified axis!'()
    {
        given :
            var a = Tsr.of(Float, [3, 1], [8, -4, 7]).setRqsGradient(true)
            var b = Tsr.of(Float).withShape(3, 1).andFill(5, -1, 2).setRqsGradient(true)
        and :
            var cat = Function.of('concat(I[0], I[1])')

        when :
            var c = cat.callWith(Arg.Axis.of(1)).call(a, b)

        then :
            c.shape() == [3, 2]

        when :
            var y = c / 2
        and :
            y.backward(Tsr.ofFloats().withShape(3,2).andFill(-1, 2, 0.5, 3, -0.1, 4))

        then :
            a.gradient.value == [-0.5, 0.25, -0.05] as float[]
            b.gradient.value == [1.0, 1.5, 2.0] as float[]
    }



}
