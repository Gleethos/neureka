package ut.ndas

import neureka.Nda
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Nda Inline Assignment")
@Narrative('''

    In this specification we cover the behaviour of nda's with respect to the assignment operation
    as well as the assignment of individual Nda items.

''')
@Subject([Nda])
class Nda_Assign_Spec extends Specification
{
    def 'We can use the "mut" API to assign the contents of one nd-array into another one.'()
    {
        given : 'We have two nd-arrays:'
            var a = Nda.of('x', 'y', 'z')
            var b = Nda.of('1', '2', '3')
        when : 'We assign the contents of "b" into "a" using the "mut" API:'
            a.mut.assign( b )
        then : 'The contents of "a" should be the same as the contents of "b":'
            a.items == b.items
    }

    def 'Assignment can be easily achieved through subscription operators.'()
    {
        given : 'An nda of ints with shape (2, 3).'
            var n = Nda.of(Integer).withShape(2, 3).andFill(1, 2, 3, 4, 5, 6)
        and : 'An nda of ints with shape (1, 2).'
            var a = Nda.of(Integer).withShape(1, 2).andFill(42, 42)

        when : 'We assign the nda a to the nda n.'
            n.mut[0, 0..1] = a
        then : 'The nda n has the expected values.'
            n.items == [42, 42, 3, 4, 5, 6]
    }

    def 'We can assign one slice into another one.'()
    {
        reportInfo """
            Using the 'assign' operation on slices should be handled with care,
            since the operation has side effects on the underlying data array
            which is shared by both the slice and its parent.
            Use the 'copy' operation on slices if you want to avoid this.
        """

        given : 'Two nd-arrays of ints with shape (5).'
            var n1 = Nda.of(Byte).vector(1, 2, 3, 4, 5)
            var n2 = Nda.of(Byte).vector(6, 7, 8, 9, 10)

        when : 'We create to very simple slices which are simply the first 3 items of the above vectors.'
            var s1 = n1[0..2]
            var s2 = n2[0..2]
        then : 'The slices will have the expected state.'
            s1.items == [1, 2, 3]
            s2.items == [6, 7, 8]

        when : 'We now assign the first slice into the second one.'
            s2.mut.assign(s1)

        then : 'Both slices will have the same numbers "1, 2, 3" in them.'
            s1.items == [1, 2, 3]
            s2.items == [1, 2, 3]
        and : 'The 2 original vectors will also both have the same numbers "1, 2, 3" in them.'
            n1.items == [1, 2, 3, 4, 5]
            n2.items == [1, 2, 3, 9, 10]
    }

}
