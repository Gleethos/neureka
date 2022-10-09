package ut.ndas

import neureka.Nda
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Nda Inline Assignment")
@Narrative('''

    In this specification we cover the behaviour of nda's with respect to the assignment operation
    as well as the assignment of individual Nda items.

''')
class Nda_Assign_Spec extends Specification
{
    def 'Assignment can be easily achieved through subscription operators.'()
    {
        given : 'An nda of ints with shape (2, 3).'
            var n = Nda.of(Integer).withShape(2, 3).andFill(1, 2, 3, 4, 5, 6)
        and :
            var a = Nda.of(Integer).withShape(1, 2).andFill(42, 42)

        when : 'We assign the nda a to the nda n.'
            n.mut[0, 0..1] = a
        then : 'The nda n has the expected values.'
            n.items == [42, 42, 3, 4, 5, 6]
    }

    def 'We can assign one slice into another one.'()
    {
        given :
            var n1 = Nda.of(Byte).vector(1, 2, 3, 4, 5)
            var n2 = Nda.of(Byte).vector(6, 7, 8, 9, 10)

        when : 'We create to very simple slices which are simply the first 3 items of the above vectors.'
            var s1 = n1[0..2]
            var s2 = n2[0..2]
        then : 'The slices will have the expected state.'
            s1.items == [1, 2, 3] as List<Byte>
            s2.items == [6, 7, 8] as List<Byte>

        when : 'We now assign the first slice into the second one.'
            s2.mut.assign(s1)

        then : 'Both slices will have the same numbers "1, 2, 3" in them.'
            s1.items == [1, 2, 3] as List<Byte>
            s2.items == [1, 2, 3] as List<Byte>
        and : 'The 2 original vectors will also both have the same numbers "1, 2, 3" in them.'
            n1.items == [1, 2, 3, 4, 5] as List<Byte>
            n2.items == [1, 2, 3, 9, 10] as List<Byte>
    }

}
