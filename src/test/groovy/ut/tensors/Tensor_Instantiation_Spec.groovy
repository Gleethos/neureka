package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Instantiating Tensors")
@Narrative('''

    Tensors are complicated data structures with a wide range of different possible states.
    They can host elements of different types residing on many kinds of different devices.
    Here we want to show how a tensor can be instantiated in different ways.
                    
''')
@Subject([Tsr])
class Tensor_Instantiation_Spec extends Specification
{
    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def 'Vector tensors can be instantiated via factory methods.'(
        def data, Class<?> type, List<Integer> shape
    ) {
        given : 'We create a vector tensor using the "of" factory method.'
            Tsr<?> t = Tsr.of(data)
            // In practise this might be something like: Tsr.of(42, 666, 73, 64)

        expect : 'The resulting tensor has the expected item type class.'
            t.itemType == type
        and : 'Also the expected shape.'
            t.shape() == shape
        and : 'The tensor has the expected data array.'
            t.unsafe.data.ref == data
            t.data == data
        and : 'The tensor is not virtual nor is it a slice... so the item array and data array contain the same values.'
            t.items == data

        where :
            data                        ||  type  | shape
            new double[]{1.1, 2.2, 3.3} || Double | [ 3 ]
            new float[]{-0.21, 543.3}   || Float  | [ 2 ]
            new boolean[]{true, false}  || Boolean| [ 2 ]
            new short[]{1, 2, 99, -123} || Short  | [ 4 ]
            new long[]{3, 8, 4, 2, 3, 0}|| Long   | [ 6 ]
            new int[]{66, 1, 4, 42, -40}|| Integer| [ 5 ]
    }

    def 'Scalar tensors can be created via static factory methods'(def data, Class<Object> type)
    {
        given : 'We make sure that the data is of the right type (based on the data table):'
            data = data.asType(type)
        and : 'We create a scalar tensor using the "of" factory method.'
            Tsr<?> t = Tsr.of(data)
            // In practise this might be something like: Tsr.of(42)
        expect : 'The resulting tensor has the expected item type class.'
            t.itemType == type
        and : 'Also the expected shape.'
            t.shape() == [ 1 ]
        and : 'The tensor has the expected data array.'
            t.unsafe.data.ref == [data] // Internal data
            t.data == [data]
        and : 'The tensor is not virtual nor is it a slice... so the item array and data array contain the same values.'
            t.items == [data]
        where :
            data  ||  type
            1.1   || Double
            -0.21 || Double
            0.1f  || Float
            -42.9 || Float
            true  || Boolean
            false || Boolean
            99    || Integer
            -123  || Integer
            3L    || Long
            8L    || Long
            1     || Short
            2     || Short
            -12   || Byte
            3     || Byte
    }

    def 'Tensors can be instantiated based on lists for both shapes and values.'()
    {
        given :
            Tsr<Integer> t = Tsr.of([2, 2], [2, 4, 4])
        expect :
            t.toString() == "(2x2):[2.0, 4.0, 4.0, 2.0]"
        when :
            t = Tsr.of([2], [3, 5, 7])
        then :
            t.toString() == "(2):[3.0, 5.0]"
            t.getItemsAs( double[].class ).length == 2
    }

    def 'Tensors can be instantiated based on arrays for both shapes and values.'()
    {
        given :
            Tsr<Double> t = Tsr.of(new int[]{2, 2}, new double[]{2, 4, 4})
        expect :
            t.toString() == "(2x2):[2.0, 4.0, 4.0, 2.0]"
        when :
            t = Tsr.of(new int[]{2}, new double[]{3, 5, 7})
        then :
            t.toString() == "(2):[3.0, 5.0]"
            t.getItemsAs( double[].class ).length == 2
    }

    def 'Tensors can be instantiated with String seed.'()
    {
        given : 'Three seeded 2D tensors are being instantiated.'
            Tsr<Double> t1 = Tsr.of([2, 3], "I am a seed! :)")
            Tsr<Double> t2 = Tsr.of([2, 3], "I am a seed! :)")
            Tsr<Double> t3 = Tsr.of([2, 3], "I am also a seed! But different. :)")

        expect : 'Equal seeds produce equal values.'
            t1.toString() == t2.toString()
            t1.toString() != t3.toString()
    }

    def 'Passing String seed to tensor produces expected values.'()
    {
        when : Tsr r = Tsr.of([2, 2], "jnrejn")
        then : r.toString().contains("0.02847, -0.69068, 0.15386, 1.81382")
        when : r = Tsr.of([2, 2], "jnrejn2")
        then : !r.toString().contains("0.02600, -2.06129, -0.48373, 0.94884")
    }


}