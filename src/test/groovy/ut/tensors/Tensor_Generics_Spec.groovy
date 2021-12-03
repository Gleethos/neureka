package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.dtype.DataType
import neureka.view.TsrStringSettings
import spock.lang.Specification

class Tensor_Generics_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
                <h2> Tensors as Containers </h2>
                <br> 
                <p>
                    Tensors do not just store numeric data.
                    They can hold anything which can be stuffed into a "Object[]" array.
                    You could even create a tensor of tensors!            
                </p>
            """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
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

        given :
            Tsr<Double> t = Tsr.newInstance()

        expect :
            t.getRepresentativeValueClass() == Neureka.get().settings().dtype().defaultDataTypeClass
        and :
            t.getValueClass() == DataType.of(Neureka.get().settings().dtype().defaultDataTypeClass).getJVMTypeClass()


    }

    def 'String tensor instance discovers expected class.'(){

        given :
            Tsr t = Tsr.of([2, 4], ["Hi", "I'm", "a", "String", "list"])

        expect :
            t.getValueClass() == String.class

        and :
            t.getRepresentativeValueClass() == String.class

        and :
            t.toString() == "(2x4):[Hi, I'm, a, String, list, Hi, I'm, a]"

    }

    def '1D tensors can be created from primitive arrays.'(
            int size, Object data, Class<?> expected
    ){

        given :
            def t = Tsr.of(data)

        expect :
            t.rank() == 1
        and :
            t.size() == size
        and :
            t.getValueClass() == expected


        where :
            size | data                      | expected
              3  | new float[]{-1f, 3f, 6f}  | Float
              4  | new int[]{1, -2 , 9, 12}  | Integer
              2  | new byte[]{42 , 73}       | Byte
              3  | new long[]{-16 , 54, 12}  | Long
              1  | new short[]{26}           | Short

    }



}
