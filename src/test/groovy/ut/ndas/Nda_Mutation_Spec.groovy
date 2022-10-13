package ut.ndas

import neureka.Nda
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Mutating ND-Arrays")
@Narrative('''

    ND-Arrays should be considered immutable, so we should prefer creating new 
    ND-Arrays from existing ones using wither methods.
    However this is not always a good idea as it can be expensive to create new
    ND-Arrays, especially if the ND-Array is very large.
    The ability to mutate ND-Arrays is therefore provided, but only
    accessible via the mutation API exposed by the `getMut()` method.

''')
class Nda_Mutation_Spec extends Specification
{
    def 'A simple vector ND-Array can be mutated using the "setItemAt" method.'()
    {
        given: 'A simple nd-array with 5 values.'
            var nda = Nda.of('i', 't', 'e', 'm', 's')

        when: 'We mutate the nd-array by setting the value at index 2 to "E".'
            nda.mut.setItemAt(2, 'E')

        then: 'The list of items now reflects the change.'
            nda.items == ['i', 't', 'E', 'm', 's']
    }

    def 'A ND-Array can be mutated simply using the "set" method.'()
    {
        reportInfo """
            This method of mutation is best used in Kotlin where it translates
            to the "set" operator.
            So it is possible to write code like this: `nda[2, 3] = 42.0`
        """

        given: 'A rank 2 nd-array with 4 values.'
            var nda = Nda.of(Byte).withShape(2, 2).andFill(1,2,3,4)

        when: 'We mutate the nd-array by setting the value at index 1 to 5.'
            nda.mut.set(1, 0, 5 as byte)

        then: 'The list of items now reflects the change.'
            nda.items == [1, 2, 5, 4]
    }

    def 'A simple vector ND-Array can be mutated using the "at(..).set(..)" methods.'()
    {
        given: 'A simple nd-array with 5 values.'
            var nda = Nda.of('i', 't', 'e', 'm', 's')

        when: 'We mutate the nd-array by setting the value at index 2 to "E".'
            nda.mut.at(2).set('E')

        then: 'The list of items now reflects the change.'
            nda.items == ['i', 't', 'E', 'm', 's']
    }

    def 'A ND-Array can be mutated using the "at(..).set(..)" methods.'()
    {
        given: 'A rank 2 nd-array with 4 values.'
            var nda = Nda.of(Byte).withShape(2, 2).andFill(1,2,3,4)

        when: 'We mutate the nd-array by setting the value at index 1 to 5.'
            nda.mut.at(1, 0).set(5 as byte)

        then: 'The list of items now reflects the change.'
            nda.items == [1, 2, 5, 4]
    }

    def 'We can use the subscription operator to mutate a simple vector ND-Array.'()
    {
        given: 'A simple nd-array with 5 values.'
            var nda = Nda.of('i', 't', 'e', 'm', 's')

        when: 'We mutate the nd-array by setting the value at index 2 to "E".'
            nda.mut[2] = 'E'

        then: 'The list of items now reflects the change.'
            nda.items == ['i', 't', 'E', 'm', 's']
    }

    def 'We can use the subscription operator to mutate an ND-Array.'()
    {
        given: 'A rank 2 nd-array with 4 values.'
            var nda = Nda.of(Byte).withShape(2, 2).andFill(1,2,3,4)

        when: 'We mutate the nd-array by setting the value at index 1 to 5.'
            nda.mut[1, 0] = 5 as byte

        then: 'The list of items now reflects the change.'
            nda.items == [1, 2, 5, 4]
    }

}
