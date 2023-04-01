package ut.tensors

import neureka.Shape
import neureka.Tsr
import spock.lang.Ignore
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Tensor Dot Products")
@Narrative('''

    A tensor can also be a simple vector, which is a tensor of rank 1.
    This specification demonstrates how to perform dot products on tensors of rank 1.
    
''')
@Subject([Tsr])
class Tensor_Dot_product_Spec extends Specification
{
    def 'The "dot" method calculates the dot product between vectors.'()
    {
        given : 'two vectors, a and b, of length 2.'
            var a = Tsr.of(3f, 2f)
            var b = Tsr.of(1f, -0.5f)
        when : 'we calculate the dot product of a and b.'
            var result = a.dot(b)
        then : 'the result is a scalar.'
            result.shape == Shape.of(1)
            result.items == [ 3f * 1f + 2f * -0.5f ]
    }

    def 'You can slice a Matrix into vectors and then used them for dot products.'()
    {
        given : 'A matrix we want to slice.'
            var m = Tsr.of(1f..4f).withShape(2,2)
        when : 'we slice the matrix into two vectors.'
            var a = m.slice().axis(0).at(0).get()
            var b = m.slice().axis(0).at(1).get()
        and : 'We perform a dot product on the two vectors.'
            var c = a.dot(b)

        then : 'the result is a scalar.'
            c.shape == Shape.of(1)
            c.items == [ 1f * 3f + 2f * 4f ]
    }

    def 'The "dot" operation supports autograd.'()
    {
        reportInfo """
            The "dot" operation supports autograd.
            This means that you can use it to calculate the gradient of a weight tensor.
            This is useful for when you want to build a neural network or some other machine learning model.
        """
        given : 'two vectors, a and b, of length 2.'
            var a = Tsr.of(4f, -1f, 2f)
            var w = Tsr.of(1f, 0f, 0f).setRqsGradient(true)
        when : 'we calculate the dot product of a and w.'
            var result = a.dot(w)
        then : 'the result is a scalar.'
            result.shape == Shape.of(1)
            result.items == [ 4f ]

        when : 'we calculate the gradient of the result with respect to w, divided by 2.'
            result.backward(0.5f)
        then : 'the gradient of w is a vector of length 3.'
            w.gradient.isPresent()
            w.gradient.get().shape == Shape.of(3)
            w.gradient.get().items == [ 2f, -0.5f, 1f ]
    }

}
