package ut.tensors

import neureka.Tsr
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Copy or Not to Copy")
@Narrative('''

    In this specification we cover the behaviour of tensors with respect to their copy methods.
    There are to main ways to copy a tensor: <br>
    1. .shallowCopy() <br>
    2. .deepCopy() <br>
    <br>
    The first method creates a new tensor with the same underlying data array as the original tensor. <br>
    The second method on the other hand creates a new tensor with a new data array. <br>
    <br>
    The first method is the most efficient, but it is not as safe as the second method. <br>
    The second method is the most safe, but it is not as efficient. <br>
    <br>
    Besides these 2 main requirements, there are als some corner cases with respect to
    the components of a tensor (like for example its computation graph) which
    will be covered in this specification as well.

''')
class Copy_Spec extends Specification
{

    def 'A deep copy of a tensor is also a deep copy of the underlying data array.'()
    {
        given : 'A tensor of ints with shape (2, 3).'
            var t = Tsr.ofInts().withShape(2, 3).andFill(1, 2, -9, 8, 3, -2)
        expect : 'The underlying data array is as expected.'
            t.unsafe.data == [1, 2 ,-9, 8, 3, -2] // It's unsafe because it exposes mutable parts of the tensor!

        when : 'We create a deep copy of the tensor.'
            var deep = t.deepCopy()
        then : 'The copy is not the same instance as the original tensor.'
            deep !== t // It's not the same instance!
        and : 'The shape and underlying data array are equal to the original tensor but the data is not identical.'
            deep.shape == t.shape
            deep.unsafe.data == t.unsafe.data // The tensors share the same values!
            deep.unsafe.data !== t.unsafe.data // ...but they are not the same array!
        and :
            (0..<t.size).every({ int i -> deep.at(i) == t.at(i) }) // The values are the same!
    }


    def 'A deep copy of a slice tensor is also a deep copy of the underlying data array.'()
    {
        given :
            var t = Tsr.ofInts().withShape(3, 3).andFill(1, 2, -9, 8, 3, -2)[0..1, 1..2]
        expect :
            t.items == [2, -9, 3, -2]
            t.data == [1, 2, -9, 8, 3, -2, 1, 2, -9]
            t.unsafe.data == [1, 2, -9, 8, 3, -2, 1, 2, -9] // It's unsafe because it exposes mutable parts of the tensor!

        when :
            var deep = t.deepCopy()
        then :
            deep.items == [2, -9, 3, -2]
            deep.data == [2, -9, 3, -2]
            deep.unsafe.data == [2, -9, 3, -2] // It's unsafe because it exposes mutable parts of the tensor!
        and :
            deep !== t // It's not the same instance!
            deep.shape == t.shape
            deep.items == t.items // The tensors share the same values!
            deep.items !== t.items // The tensors share the same values!
            deep.unsafe.data !== t.unsafe.data // The tensors share the same values!
            deep.unsafe.data !== t.unsafe.data // ...but they are not the same array!
        and :
            (0..<t.size).every({ int i -> deep.at(i) == t.at(i) }) // The values are the same!
    }

    def 'A shallow copy will share the same underlying data as its original tensor.'()
    {
        given :
            var t = Tsr.ofInts().withShape(2, 3).andFill(1, 2, -9, 8, 3, -2)
        expect :
            t.unsafe.data == [1, 2 ,-9, 8, 3, -2] // It's unsafe because it exposes mutable parts of the tensor!

        when :
            var shallow = t.shallowCopy()
        then :
            shallow !== t // It's not the same instance!
            shallow.shape == t.shape
            shallow.unsafe.data == t.unsafe.data // The tensors share the same values!
            shallow.unsafe.data === t.unsafe.data // The tensors share the exact same data array!
        and :
            (0..<t.size).every({ int i -> shallow.at(i) == t.at(i) }) // The values are the same!
    }

}
