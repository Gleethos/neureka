package ut.tensors

import neureka.Shape
import neureka.Tensor
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("The 'dimTrim' Method")
@Narrative('''
    The 'dimTrim' method is used to remove training and leading dimensions of length 1 from a tensor.
    This is useful when you want to perform operations on tensors of different ranks.
    For example, if you want to perform a dot product on two vectors, you can use the 'dimTrim' method
    to remove the dimension of length 1 from the vector, so that it becomes a scalar.
    This way you can perform the dot product on two scalars.
''')
@Subject([Tensor])
class DimTrim_Spec extends Specification
{

    def 'The "dimTrim" operation works on slices too!'()
    {
        given : 'A matrix we want to slice.'
            var m = Tensor.of(1f..4f).reshape(2,2)
        when : 'we slice the matrix into two vectors.'
            var a = m.slice().axis(0).at(0).get()
            var b = m.slice().axis(0).at(1).get()
        and : 'We apply the "dimTrim" operation on the two vectors.'
            a = a.dimtrim()
            b = b.dimtrim()
        then : 'the result is a vector of length 2.'
            a.shape == Shape.of(2)
            b.shape == Shape.of(2)
        and : 'They have the same items as the original vectors.'
            a.items == [ 1f, 2f ]
            b.items == [ 3f, 4f ]
    }

}
