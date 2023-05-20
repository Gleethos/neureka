package ut.tensors

import neureka.Tensor
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Tensor Inline Assignment")
@Narrative('''

    In this specification we cover the behaviour of tensors with respect to the assignment operation
    as well as the assignment of individual tensor items.

''')
@Subject([Tensor])
class Tensor_Assign_Spec extends Specification
{
    def 'We can use the "mut" API to assign the contents of one tensor into another one.'()
    {
        given : 'We have two vector tensors:'
            var a = Tensor.of(1L, 2L, 3L)
            var b = Tensor.of(-3L, -2L, -1L)
        when : 'We assign the contents of "b" into "a" using the "mut" API:'
            a.mut.assign( b )
        then : 'The contents of "a" should be the same as the contents of "b":'
            a.items == b.items
    }

    def 'Assignment can be easily achieved through subscription operators.'()
    {
        given : 'An tensor of bytes with shape (2, 3).'
            var n = Tensor.ofBytes().withShape(3, 2).andFill(5, 4, 3, 2, 1, 0)
        and :
            var a = Tensor.ofBytes().withShape(1, 2).andFill(-42, 42)

        when : 'We assign the tensor a to the tensor n.'
            n.mut[0..1, 1] = a
        then : 'The Tensor n has the expected values.'
            n.items == [5, -42, 3, 42, 1, 0]
    }

    def 'We can assign one slice into another one.'()
    {
        reportInfo """
            Note that using the 'assign' operation on slices should be handled with care,
            since the operation has side effects on the underlying data array
            which is shared by both the slice and its parent.
            Use the 'copy' operation on slices if you want to avoid this.
        """
        given :
            var n1 = Tensor.ofShorts().vector(1, 2, 3, 4)
            var n2 = Tensor.ofShorts().vector(6, 7, 8, 9, 10, 11)

        when : 'We create to very simple slices of 3 items in the above vectors.'
            var s1 = n1[0..2]
            var s2 = n2[2..4]
        then : 'The slices will have the expected state.'
            s1.items == [1, 2, 3]
            s2.items == [8, 9, 10]

        when : 'We now assign the first slice into the second one.'
            s2.mut.assign(s1)

        then : 'Both slices will have the same numbers "1, 2, 3" in them.'
            s1.items == [1, 2, 3]
            s2.items == [1, 2, 3]
        and : 'The 2 original vectors will also both have the same numbers "1, 2, 3" in them.'
            n1.items == [1, 2, 3, 4]
            n2.items == [6, 7, 1, 2, 3, 11]
    }

}
