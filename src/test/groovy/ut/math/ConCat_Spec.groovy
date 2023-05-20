package ut.math

import neureka.Neureka
import neureka.Tensor
import neureka.math.Function
import neureka.math.args.Arg
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Merging Tensors")
@Narrative('''

    Tensors can not only be sliced, but also merged.
    This is most easily achieved through the concatenation operation, 
    which stacks 2 tensors alongside a specified axis.
    This specification not only covers how you can concatenate tensors,
    but also how this works alongside autograd and non-numeric tensors.

''')
class ConCat_Spec extends Specification
{

    def 'We can concatenate 2 tensors alongside a specified axis!'()
    {
        given : 'We create 2 rank 3 tensors, which we want to concatenate, where the first one requires gradients.'
            var a = Tensor.of(Double, [3, 4, 2], [-1.7, 2, 9.3, -3]).setRqsGradient(true)
            var b = Tensor.ofDoubles().withShape(3, 2, 2).andFill(3,2.5,-6)
        and : 'A function which should perform the concatenation.'
            var cat = Function.of('concat(I[0], I[1])')

        when : 'We call the previously created function alongside the axis alongside we want to concatinate '
            var c = cat.with(Arg.Axis.of(1)).call(a, b)

        then : 'The resulting tensor should have the expected shape.'
            c.shape() == [3, 6, 2]

        when : 'We use the result for some more operations...'
            var y = c * 2
        and : 'We back-propagate -3 on y...'
            y.backward(-3)

        then : 'The gradient of the first tensor should look as follows!'
            a.gradient.get().every(it -> it == -6 )
    }


    def 'We can concatenate 2 float tensors alongside a specified axis!'()
    {
        given : 'We create 2 rank 2 tensors, which we want to concatenate, where both require gradients.'
            var a = Tensor.of(Float, [4, 2], [8f, -9f, 5f]).setRqsGradient(true)
            var b = Tensor.ofFloats().withShape(4, 3).andFill(1,6,-6,3).setRqsGradient(true)
        and : 'A function which should perform the concatenation.'
            var cat = Function.of('concat(I[0], I[1])')

        when : 'We call the previously created function with the axis alongside we want to concatenate.'
            var c = cat.with(Arg.Axis.of(1)).call(a, b)

        then : 'The resulting tensor should have the expected shape.'
            c.shape() == [4, 5]

        when : 'We perform some more operations on top of the previous concatenation...'
            var y = c * 5 + 1
        and : 'Then we trigger autograd on the most recent result...'
            y.backward(-2)

        then : 'The original leave tensors used for the merging have received the expected gradients.'
            a.gradient.get().every(it -> it == -10 )
            b.gradient.get().every(it -> it == -10 )
    }


    def 'We can concatenate 2 string tensors alongside a specified axis!'()
    {
        given : 'We create 2 rank 2 string tensors, which we want to concatenate, where both require gradients.'
            var a = Tensor.of(String, [2, 5], [':)', ':P', 'B)'])
            var b = Tensor.of(String).withShape(1, 5).andFill('O.o', '._.')
        and : 'A function which should perform the concatenation.'
            var cat = Function.of('concat(I[0], I[1])')

        when : 'We call the previously created function alongside the axis alongside we want to concatenate.'
            var c = cat.with(Arg.Axis.of(0)).call(a, b)

        then : 'The resulting tensor should have the expected shape.'
            c.shape() == [3, 5]
        and :
            c.any( it -> it == ':P' )
            c.any( it -> it == '._.' )
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null })
    def 'We can concatenate and then back-propagate 2 simple float tensors alongside a specified axis!'(
        Device<?> device
    ) {
        given : 'We create 2 rank 2 tensors, which we want to concatenate, where both require gradients.'
            var a = Tensor.of(Float, [3, 1], [8, -4, 7]).setRqsGradient(true)
            var b = Tensor.of(Float).withShape(3, 1).andFill(5, -1, 2).setRqsGradient(true)
        and : 'A function which should perform the concatenation.'
            var cat = Function.of('concat(I[0], I[1])')

        when : 'We call the previously created function alongside the axis alongside we want to concatenate.'
            var c = cat.with(Arg.Axis.of(1)).call(a, b)

        then : 'The resulting tensor should have the expected shape.'
            c.shape() == [3, 2]

        when :
            var y = c / 2
        and :
            y.backward(Tensor.ofFloats().withShape(3,2).andFill(-1, 2, 0.5, 3, -0.1, 4))

        then :
            a.gradient.get().items == [-0.5, 0.25, -0.05] as float[]
            b.gradient.get().items == [1.0, 1.5, 2.0] as float[]

        where :
            device << [CPU.get(), Device.get(OpenCLDevice, 'gpu')]
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null })
    def 'We can concatenate and then back-propagate 3 simple float tensors alongside a specified axis!'(
            Device<?> device
    ) {
        given : 'We create 2 rank 2 tensors, which we want to concatenate, where both require gradients.'
            var a = Tensor.of(Float, [1, 3], [8, -4, 7]).setRqsGradient(true)
            var b = Tensor.of(Float).withShape(1, 3).andFill(5, -1, 2).setRqsGradient(true)
            var c = Tensor.ofRandom(Float, 1, 3).setRqsGradient(true)
        and : 'A function which should perform the concatenation.'
            var cat = Function.of('concat(I[0], I[1], I[2])')

        when : 'We call the previously created function alongside the axis alongside we want to concatenate.'
            var d = cat.with(Arg.Axis.of(0)).call(a, b, c)

        then : 'The resulting tensor should have the expected shape.'
            d.shape() == [3, 3]

        when :
            var y = d ** 2
        and :
            y.backward(Tensor.ofFloats().withShape(3,3).andFill(-1, 2, 0.5, 3, -0.1, 4))

        then :
            a.gradient.get().items == [-16, -16, 7] as float[]
            b.gradient.get().items == [30, 0.2, 16] as float[]
            c.gradient.get().items == [0.30829078, -3.1254156, -0.52700233] as float[]

        where :
            device << [CPU.get(), Device.get(OpenCLDevice, 'gpu')]
    }


}
