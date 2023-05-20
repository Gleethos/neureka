package ut.tensors

import neureka.Neureka
import neureka.Tensor
import neureka.dtype.DataType
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Tensors as Generic Containers")
@Narrative('''

    Tensors do not just store numeric data.
    They can hold anything which can be stuffed into a "Object[]" array.
    You could even create a tensor of tensors!  

''')
class Tensor_Generics_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
                    Here you will find out how to create a tensor of any kind of data.          
            """
    }

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

    def 'Anonymous tensor instance has the default datatype class as defined in Neureka settings.'() {

        given : 'We create a completely uninitialized tensor instance.'
            var t = Tensor.newInstance()

        expect :
            t.getRepresentativeItemClass() == Neureka.get().settings().dtype().defaultDataTypeClass
        and :
            t.getItemType() == DataType.of(Neureka.get().settings().dtype().defaultDataTypeClass).getItemTypeClass()
    }

    def 'We can create a tensor of strings.'()
    {
        given : 'We create a tensor of strings.'
        Tensor<String> t = Tensor.of([2, 4], ["Hi", "I'm", "a", "String", "list"])

        expect : 'The tensor has the correct item type.'
            t.itemType == String.class

        and :
            t.getRepresentativeItemClass() == String.class

        and :
            t.toString() == "(2x4):[Hi, I'm, a, String, list, Hi, I'm, a]"
    }

    def '1D tensors can be created from primitive arrays.'(
            int size, Object data, Class<?> expected
    ){
        given :
            def t = Tensor.of(data)

        expect :
            t.rank() == 1
        and :
            t.size() == size
        and :
            t.getItemType() == expected


        where :
            size | data                      | expected
              3  | new float[]{-1f, 3f, 6f}  | Float
              4  | new int[]{1, -2 , 9, 12}  | Integer
              2  | new byte[]{42 , 73}       | Byte
              3  | new long[]{-16 , 54, 12}  | Long
              1  | new short[]{26}           | Short
    }



}
