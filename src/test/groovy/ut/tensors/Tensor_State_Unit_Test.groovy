package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.dtype.DataType
import neureka.dtype.custom.I8
import spock.lang.Specification

class Tensor_State_Unit_Test extends Specification
{
    def setupSpec() {
        reportHeader """    
            <h2>Tensor State Tests</h2>
            <p>
                This unit test specification covers the expected state of newly instantiated tensors.
                Certain properties must have their expected default values.
            </p>
        """
    }

    def 'Newly instantiated and unmodified scalar tensor has expected state.'()
    {
        given: 'A new instance of a scalar tensor.'
            Tsr t = new Tsr( 6 )
        expect: 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
        and : 'The tensor contains the expected data.'
            t.value64() == [6] as double[]
            t.value32() == [6] as float[]
            t.data == [6] as double[]
            t.value == [6] as double[]

        when: 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
        then: 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !t.is64() && !t.is32()
            t.value64() == null
            t.value32() == null
            t.data == null
            t.value == null
        when: 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then: 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.value64() == [0] as double[]
            t.value32() == [0] as float[]
            t.data == [0] as double[]
            t.value == [0] as double[]
            t.isVirtual()

    }

    def 'Newly instantiated and unmodified vector tensor has expected state.'()
    {
        given : 'A new vector tensor is being instantiated.'
            Tsr t = new Tsr( new int[]{ 2 }, 5 )
        expect : 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
        when : 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
        then : 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !t.is64() && !t.is32()
            t.dataType.getTypeClass() == Neureka.instance().settings().dtype().defaultDataTypeClass
            t.value64() == null
            t.value32() == null
            t.data == null
            t.value == null
        when : 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then : 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.value64() == [0, 0] as double[]
            t.value32() == [0, 0] as float[]
            t.data == [0] as double[]
            t.value == [0, 0] as double[]
            t.isVirtual()
    }


    def 'Tensor created from shape and datatype has expected state.'()
    {
        given : 'A new vector tensor is being instantiated.'
            Tsr t = new Tsr( new int[]{ 2 }, DataType.instance(I8.class ) )
        expect : 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
            t.value64() == [0, 0] as double[]
            t.value32() == [0, 0] as float[]
            t.data == [0] as byte[]
            t.value == [0, 0] as byte[]
            t.isVirtual()
        when : 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
        then : 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !t.is64() && !t.is32()
            t.dataType.getTypeClass() == I8.class
            t.value64() == null
            t.value32() == null
            t.data == null
            t.value == null
        when : 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then : 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.value64() == [0, 0] as double[]
            t.value32() == [0, 0] as float[]
            t.data == [0] as byte[]
            t.value == [0, 0] as byte[]
            t.isVirtual()
    }

}
