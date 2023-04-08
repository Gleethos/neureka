package ut.tensors


import neureka.Tsr
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Tensor Reshaping")
@Narrative('''

    This specification demonstrates how to reshape tensors,
    which means to change the shape of a tensor.

    Note that immutability is a core concept of the Neureka library.
    This means that the `Tsr` API does not expose mutability directly.
    Instead, it exposes methods that return new instances of `Tsr`
    that are derived from the original instance.
    
    This is also true for reshaping operations.
    
    Don't be concerned about the performance implications of this,
    because in the vast majority of cases the new instance will be backed by the same data array
    as the original instance!
    
''')
@Subject([Tsr])
class Reshape_Spec extends Specification
{
    def 'We can create a new tensor with a different shape.'()
    {
        given : 'We create a new tensor with a shape of [2, 3].'
            Tsr<?> t = Tsr.of( 6..12 ).withShape( 2, 3 )
        expect : 'The new instance will have the expected shape.'
            t.shape() == [2, 3]
        and : 'The new instance will have the expected items.'
            t.items() == [6, 7, 8, 9, 10, 11]
        and : 'The new instance will have the same data type as the original instance.'
            t.itemType() == Integer
    }

    def 'The reshape operation supports autograd!'()
    {
        reportInfo """
            Changing the shape of a tensor is a very common operation in machine learning.
            This is why the reshape operation also supports autograd.
            So for example when you have a tensor `a` with shape `s1` and you reshape it to 
            a new tensor `b` with shape `s2` then during backpropagation the error `e_b` of `b`
            with the shape `s2` will be propagated to a new error `e_a` of `a` with the shape `s1`.
            It is basically the reshape operation applied in reverse.
        """
        given : 'We create a new tensor with a shape of [3, 2] that requires gradients (so that we can capture the error).'
            Tsr<?> a = Tsr.of( 1..6 ).withShape( 3, 2 ).setRqsGradient( true )
        when : 'We reshape the tensor to a new tensor with shape [2, 3].'
            Tsr<?> b = a.withShape( 2, 3 )
        then : 'The new tensor will have the expected shape.'
            b.shape() == [2, 3]

        when : 'We back-propagate an error of some random numbers...'
            b.backward( Tsr.of( -1, 3, 42, 6, -3, 9 ).withShape( 2, 3 ) )
        then : '... the error of the original tensor will have the expected shape.'
            a.gradient().get().shape() == [3, 2]
        and : '... the error of the original tensor will have the expected items.'
            a.gradient().get().items() == [-1, 3, 42, 6, -3, 9]
    }
}
