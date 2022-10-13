package ut.ndas

import neureka.Nda
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Nda Withers")
@Narrative('''

    Immutability is a core concept of the Neureka library.
    This means that the Nda API does not expose mutability directly.
    Instead, the API exposes methods that return new instances of Nda
    that are derived from the original instance.
    
''')
@Subject([Nda])
class Nda_Wither_Specification extends Specification
{
    def 'We can create a new Nda instance with a different shape.'()
    {
        given : 'We create a new Nda instance with a shape of [3, 2].'
            Nda<?> nda = Nda.of( 1, 2, 3, 4, 5, 6 ).withShape( 3, 2 )
        expect : 'The new instance will have the expected shape.'
            nda.shape() == [3, 2]
        and : 'The new instance will have the expected items.'
            nda.items() == [1, 2, 3, 4, 5, 6]
        and : 'The new instance will have the same data type as the original instance.'
            nda.itemType() == Integer
    }

    def 'An Nda can be labeled.'()
    {
        given : 'We create a vector of Strings.'
            Nda<?> nda = Nda.of( "a", "b", "c" )
        expect : 'Initially the vector is not labeled.'
            nda.label == ""
        when : 'We label the vector.'
            nda = nda.withLabel( "my-label" )
        then : 'The vector will have the expected label.'
            nda.label == "my-label"
    }

}
