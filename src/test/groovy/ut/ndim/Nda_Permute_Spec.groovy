package ut.ndim

import neureka.Nda
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Reshaping Nd-Arrays")
@Narrative('''

    Permuting an N-dimensional array means rearranging the dimensions/axes of the N-dimensional array.
    It returns a new nd-array with the same data as the original nd-array, 
    but with the specified dimensions rearranged. 
    It is very useful for example when you want to
    change the order of dimensions, for example, if you have a nd-array with dimensions (batch_size, channels, height, width), 
    you can use permute() to rearrange the dimensions to (batch_size, height, width, channels).
    Another useful application of permute() is transposing a matrix.
    For example, if you have a matrix with dimensions (rows, columns), 
    you can use permute() to rearrange the dimensions to (columns, rows).
 
    Permuting is a very cheap operation because it does not copy any data but merely
    creates a new view on the same data with a different access pattern.
''')
@Subject([Nda])
class Nda_Permute_Spec extends Specification
{
    def 'We can use the "permute" method to rearrange the dimensions of an nd-array.'()
    {
        reportInfo """
            In Neureka `Nda::permute(int...)` rearranges the original nd-array according to the desired 
            ordering and returns a new multidimensional rotated nd-array. 
            The size of the returned nd-array remains the same as that of the original.
        """
        given : 'A nd-array with a shape of [7, 2, 4, 3]'
            Nda t = Nda.of(-12..11).withShape(7, 2, 4, 3)

        when : 'We create a new permuted nd-array with the shape of [4, 3, 7, 2] and store it as `t2`.'
            var t2 = t.permute( 2, 3, 0, 1 )

        then : 'The new nd-array has the shape of [4, 3, 7, 2].'
            t2.shape == [4, 3, 7, 2]
    }

    def 'We can use the "transpose" method to transpose swap 2 dimensions.'()
    {
        reportInfo """
            The `transpose` method is a special case of the `permute` method
            which only swaps 2 dimensions (instead of all of them).
            It is based on the algorithm of the `permute` method.
        """
        given : 'A nd-array with a shape of [2, 1, 4, 5]'
            Nda t = Nda.of(-12..11).withShape(2, 1, 4, 5)

        when : 'We create a new transposed nd-array where the 2nd and 3rd dimensions are swapped and store it as `t2`.'
            var t2 = t.transpose( 1, 2 )

        then : 'The new nd-array has the shape of [2, 4, 1, 5].'
            t2.shape == [2, 4, 1, 5]
    }

}
