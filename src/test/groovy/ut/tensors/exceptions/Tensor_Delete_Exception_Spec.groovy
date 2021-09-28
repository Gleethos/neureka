package ut.tensors.exceptions

import neureka.Tsr
import neureka.dtype.DataType
import spock.lang.Specification

class Tensor_Delete_Exception_Spec extends Specification
{

    def 'A deleted tensor will tell you that it has been deleted.'() {

        given : 'We create a scalar tensor and then immediately delete it.'
            Tsr t = Tsr.of(-2).delete()

        expect: 'This tensor will tell you that it is deleted through the "toString" method.'
            t.toString() == "deleted"
    }

    def 'A deleted tensor will throw an exception when accessing its configuration.' () {

        given : 'We create a tensor and immediately delete it.'
            Tsr t = Tsr.of(new int[]{1, 2}, -2..4).delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to access the NDConfiguration instance...'
            t.getNDConf()

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to access the ND-Configuration of an already deleted tensor."
    }


    def 'A deleted tensor will throw an exception when trying to set its configuration.' () {

        given : 'We create a tensor and immediately delete it.'
            Tsr t = Tsr.of(new int[]{1, 2}, -2..4).delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to access the NDConfiguration instance...'
            t.setNDConf(null)

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to set the ND-Configuration of an already deleted tensor."

    }


    def 'A deleted tensor will throw an exception when accessing its data.' () {

        given : 'We create a tensor and immediately delete it.'
            Tsr t = Tsr.of(new int[]{2, 1}, -3..2).delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to access the data object...'
            t.getData()

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to access the data object of an already deleted tensor."

    }


    def 'A deleted tensor will throw an exception when trying to modify its data.' () {

        given : 'We create a tensor and immediately delete it.'
            Tsr t = Tsr.of(new int[]{2}, -3..2).delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to access the data object...'
            t.setDataAt(0, 7)

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to modify the data object of an already deleted tensor."

    }


    def 'A deleted tensor will throw an exception when accessing its data type.' () {

        given : 'We create a tensor and immediately delete it.'
            Tsr t = Tsr.of(new int[]{2, 1}, -3..2).delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to access the DataType instance...'
            t.getDataType()

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to access the data type of an already deleted tensor."
    }


    def 'A deleted tensor will throw an exception when modifying its data type.' () {

        given : 'We create a tensor and immediately delete it.'
            Tsr t = Tsr.of(new int[]{2, 1}, -3..2).delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to access the DataType instance...'
            t.setDataType(DataType.of(Float.class))

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to set the data type of an already deleted tensor."
    }


}
