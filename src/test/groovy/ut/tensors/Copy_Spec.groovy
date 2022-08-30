package ut.tensors

import neureka.Tsr
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("To Copy or Not to Copy")
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
@Subject([Tsr])
class Copy_Spec extends Specification
{

    def 'A deep copy of a tensor is also a deep copy of the underlying data array.'()
    {
        given : 'A tensor of ints with shape (2, 3).'
            var t = Tsr.ofInts().withShape(2, 3).andFill(1, 2, -9, 8, 3, -2)
        expect : 'The underlying data array is as expected.'
            t.unsafe.dataArray.ref == [1, 2 ,-9, 8, 3, -2] // It's unsafe because it exposes mutable parts of the tensor!

        when : 'We create a deep copy of the tensor.'
            var deep = t.deepCopy()
        then : 'The copy is not the same instance as the original tensor.'
            deep !== t // It's not the same instance!
        and : 'The shape and underlying data array are equal to the original tensor but the data is not identical.'
            deep.shape == t.shape
            deep.unsafe.dataArray.ref == t.unsafe.dataArray.ref // The tensors share the same values!
            deep.unsafe.dataArray.ref !== t.unsafe.dataArray.ref // ...but they are not the same array!
        and :
            (0..<t.size).every({ int i -> deep.at(i) == t.at(i) }) // The values are the same!
    }


    def 'A deep copy of a slice tensor is also a deep copy of the underlying data array.'()
    {
        given : 'A slice of ints with shape (2, 2) sliced in-place from a tensor of shape (3, 3).'
            var s = Tsr.ofInts().withShape(3, 3).andFill(1, 2, -9, 8, 3, -2)[0..1, 1..2]
        expect : 'The underlying items and data array is as expected.'
            s.items == [2, -9, 3, -2]
            s.data == [1, 2, -9, 8, 3, -2, 1, 2, -9]
            s.unsafe.dataArray.ref == [1, 2, -9, 8, 3, -2, 1, 2, -9] // It's unsafe because it exposes mutable parts of the tensor!

        when : 'We create a deep copy of the tensor.'
            var deep = s.deepCopy()
        then : 'The copy is not the same instance as the original tensor.'
            deep !== s // It's not the same instance!
        and : 'The underlying items and data array are as expected.'
            deep.items == [2, -9, 3, -2]
            deep.data == [2, -9, 3, -2]
            deep.unsafe.dataArray.ref == [2, -9, 3, -2] // It's unsafe because it exposes mutable parts of the tensor!
        and : 'The slice and the copy have the same shape.'
            deep.shape == s.shape
            deep.items == s.items // The tensors share the same values!
            deep.items !== s.items // The tensors share the same values!
            deep.unsafe.dataArray.ref !== s.unsafe.dataArray.ref // The tensors share the same values!
            deep.unsafe.dataArray.ref !== s.unsafe.dataArray.ref // ...but they are not the same array!
        and : 'We verify that they share the same ints through the "every" method.'
            (0..<s.size).every({ int i -> deep.at(i) == s.at(i) }) // The values are the same!
    }

    def 'A shallow copy will share the same underlying data as its original tensor.'(Closure<Tsr> cloner)
    {
        given : 'A tensor of ints with shape (2, 3).'
            var t = Tsr.ofInts().withShape(2, 3).andFill(1, 2, -9, 8, 3, -2)
        expect : 'The underlying data array is as expected.'
            t.unsafe.dataArray.ref == [1, 2 ,-9, 8, 3, -2] // It's unsafe because it exposes mutable parts of the tensor!

        when : 'We create a shallow copy of the tensor.'
            var shallow = cloner(t)
        then : 'The copy is not the same instance as the original tensor.'
            shallow !== t // It's not the same instance!
            shallow.shape == t.shape
            shallow.unsafe.dataArray.ref == t.unsafe.dataArray.ref // The tensors share the same values!
            shallow.unsafe.dataArray.ref === t.unsafe.dataArray.ref // The tensors share the exact same data array!
        and : 'We verify that they share the same ints through the "every" method.'
            (0..<t.size).every({ int i -> shallow.at(i) == t.at(i) }) // The values are the same!

        where :
            cloner << [{Tsr x -> x.shallowCopy()},{Tsr x -> x.shallowClone()}]
    }

    def 'We can deep copy various types of tensors.'(
        Class<?> type, Object expected
    ) {
        given : 'A simple vector tensor which we are going to copy.'
            var t = Tsr.of(type).withShape(expected.length).andFill(expected)
        and : 'A slice of the tensor, which we should also be able to deep copy.'
            var s = t[1..<(expected.length - 1)]

        when :
            var deep = t.deepCopy()
            var deepSlice = s.deepCopy()
        then : 'The copy is not the same instance as the original tensor.'
            deep !== t // It's not the same instance!
            deepSlice !== s // It's not the same instance!

        and : 'The shape and underlying data array are equal to the original tensor but the data is not identical.'
            deep.shape == t.shape
            deep.unsafe.dataArray.ref == t.unsafe.dataArray.ref // The tensors share the same values!
            deep.unsafe.dataArray.ref !== t.unsafe.dataArray.ref // ...but they are not the same array!

        and : 'Both the copied tensor and its slice have the expected values.'
            deep.items == expected
            deepSlice.items == expected[1..<(expected.length - 1)]

        where : 'We can use the following types and values for the above code.'
            type     ||  expected
            Integer  ||  [6, 2, 0, -387, 22, 53, -92] as int[]
            Byte     ||  [-1, 4, 2, -49, 2, -72, 235, 0, 3] as byte[]
            Short    ||  [65, -20, -7, -8, -3, -4, -5, -6, -9] as short[]
            Long     ||  [0, 5462, -976, -3, -42, -35, -3436, -7, -89] as long[]
            Float    ||  [0.5076, -1.0, -2.4, -3.0, -4.0, -5.0, -6.0] as float[]
            Double   ||  [4.26434, -4.0, 5.3, -6.6, -7.0, 9.67] as double[]
            Boolean  ||  [true, false, true, false, true, false, true, false, true] as boolean[]
            Character||  ['t', 'e', 's', 't', 'd', 'a', 't', 'a', '!'] as char[]
            String   ||  ["test", "data", "!"] as String[]
            Object   ||  ["What", 4, 'm' as char, 1] as Object[]
    }

}
