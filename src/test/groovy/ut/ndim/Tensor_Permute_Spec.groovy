package ut.ndim

import neureka.Tsr
import neureka.ndim.config.NDConfiguration
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Reshaping Tensors")
@Narrative('''

    Permuting a tensor means rearranging the shape of a tensor.
    This is very useful for example when you want to perform a matrix multiplication on a tensor which is not a matrix.
    In this case you can permute the tensor to a matrix and then perform the multiplication.

    Permuting a tensor is also very useful when you want to perform other kinds of linear
    operations like for example doing 4D convolution with a tensor which is not compatible with 4D convolution.
    In this case you can permute to a 4D tensor and then perform the convolution.
  
    Permuting is also a very cheap operation because it does not copy any data but merely
    creates a new view on the same data with a different access pattern.
    
''')
@Subject([Tsr])
class Tensor_Permute_Spec extends Specification {

    def 'When matrices are transpose, they will change their layout type.'()
    {
        given :
            Tsr t = Tsr.ofFloats().withShape(3, 4).andSeed(42)

        expect :
            t.NDConf.layout == NDConfiguration.Layout.ROW_MAJOR

        when :
            t = t.T

        then :
            t.NDConf.layout == NDConfiguration.Layout.COLUMN_MAJOR

    }

}
