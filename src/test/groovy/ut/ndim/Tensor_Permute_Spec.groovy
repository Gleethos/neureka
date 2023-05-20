package ut.ndim


import neureka.Tensor
import neureka.ndim.config.NDConfiguration
import neureka.ndim.config.NDTrait
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Reshaping Tensors")
@Narrative('''

    Permuting an N-dimensional array means rearranging the dimensions/axes of the N-dimensional array.
    It produces a new tensor with the same data as the original tensor, 
    but with the specified dimensions rearranged. 
    
    This is very useful for example when you want to
    change the order of dimensions, for example, if you have a tensor with dimensions (batch_size, channels, height, width), 
    you can use permute() to rearrange the dimensions to (batch_size, height, width, channels).
    Another useful application of permute() is transposing a matrix.
    For example, if you have a matrix with dimensions (rows, columns), 
    you can use permute() to rearrange the dimensions to (columns, rows).
 
    Permuting is a very cheap operation because it does not copy any data but merely
    creates a new view on the same data with a different access pattern.
    
''')
@Subject([Tensor])
class Tensor_Permute_Spec extends Specification
{
    def 'We can use the "permute" method to rearrange the dimensions of a tensor.'()
    {
        reportInfo """
            In Neureka `Tensor::permute(int...)` rearranges the original tensor according to the desired 
            ordering and returns a new multidimensional rotated tensor. 
            The size of the returned tensor remains the same as that of the original.
        """
        given : 'A tensor with a shape of [2, 4, 6, 8]'
            Tensor t = Tensor.ofFloats().withShape(2, 4, 6, 8).andSeed(42)

        expect : 'By default, the tensor has a row major layout.'
            t.NDConf.layout == NDConfiguration.Layout.ROW_MAJOR
            t.NDConf.traits == [NDTrait.COMPACT, NDTrait.SIMPLE, NDTrait.ROW_MAJOR, NDTrait.CONTINUOUS_MATRIX]

        when : 'We create a new permuted tensor with the shape of [6, 4, 8, 2] and store it as `t2`.'
            var t2 = t.permute( 2, 1, 3, 0 )

        then : 'The new tensor has the shape of [6, 4, 8, 2].'
            t2.shape == [6, 4, 8, 2]
        and : 'A unspecific layout is assigned to the new tensor.'
            t2.NDConf.layout == NDConfiguration.Layout.UNSPECIFIC
            t2.NDConf.traits == [NDTrait.COMPACT, NDTrait.COL_MAJOR, NDTrait.CONTINUOUS_MATRIX]
        and : 'The new tensor has the same size as the original tensor, but it is not the same object.'
            t2.size == t.size
            t2 !== t
    }

    def 'When matrices are transpose, they will change their layout type as expected.'()
    {
        given :
            Tensor t = Tensor.ofFloats().withShape(3, 4).andSeed(42)

        expect :
            t.NDConf.layout == NDConfiguration.Layout.ROW_MAJOR
            t.NDConf.traits == [NDTrait.COMPACT, NDTrait.SIMPLE, NDTrait.ROW_MAJOR, NDTrait.CONTINUOUS_MATRIX]

        when :
            t = t.T

        then :
            t.NDConf.layout == NDConfiguration.Layout.COLUMN_MAJOR
            t.NDConf.traits == [NDTrait.COMPACT, NDTrait.COL_MAJOR, NDTrait.CONTINUOUS_MATRIX]
    }

}
