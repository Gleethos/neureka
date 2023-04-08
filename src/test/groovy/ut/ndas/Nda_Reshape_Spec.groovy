package ut.ndas

import neureka.Nda
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Nda Reshaping")
@Narrative('''

    Immutability is a core concept of the Neureka library.
    This means that the Nda API does not expose mutability directly.
    Instead, the API exposes methods that return new instances of Nda
    that are derived from the original instance.
    
    This is also true for reshaping operations, 
    meaning that the Nda API does not expose methods that mutate the shape of an Nda
    but instead provides methods that return new instances of Nda
    with a different shape.
    
    Don't be concerned about the performance implications of this,
    because in the vast majority of cases the new instance will be backed by the same data array
    as the original instance!
    
''')
@Subject([Nda])
class Nda_Reshape_Spec extends Specification
{
    def 'We can create a new Nda instance with a different shape.'()
    {
        given : 'We create a new Nda instance with a shape of [3, 2].'
            Nda<?> nda = Nda.of( 1..6 ).withShape( 3, 2 )
        expect : 'The new instance will have the expected shape.'
            nda.shape() == [3, 2]
        and : 'The new instance will have the expected items.'
            nda.items() == [1, 2, 3, 4, 5, 6]
        and : 'The new instance will have the same data type as the original instance.'
            nda.itemType() == Integer
    }

}
