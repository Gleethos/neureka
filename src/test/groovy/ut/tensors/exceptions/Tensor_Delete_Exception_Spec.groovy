package ut.tensors.exceptions

import neureka.Shape
import neureka.Tensor
import spock.lang.Specification

class Tensor_Delete_Exception_Spec extends Specification
{

    def 'A deleted tensor will tell you that it has been deleted.'() {

        given : 'We create a scalar tensor and then immediately delete it.'
            var t = Tensor.of(-2d).getMut().delete()

        expect: 'This tensor will tell you that it is deleted through the "toString" method.'
            t.toString() == "deleted"
    }

    def 'A deleted tensor will throw an exception when accessing its configuration.' () {

        given : 'We create a tensor and immediately delete it.'
            Tensor t = Tensor.of(Shape.of(1, 2), -2..4).getMut().delete()

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
            Tensor t = Tensor.of(Shape.of(1, 2), -2..4).getMut().delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to access the NDConfiguration instance...'
            t.mut.setNDConf(null)

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to access the unsafe API of an already deleted tensor."

    }


    def 'A deleted tensor will throw an exception when accessing its data.' () {

        given : 'We create a tensor and immediately delete it.'
        Tensor t = Tensor.of(Shape.of(2, 1), -3..2).getMut().delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to access the data object...'
            t.mut.data.get()

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to access the unsafe API of an already deleted tensor."

    }


    def 'A deleted tensor will throw an exception when trying to modify its data.' () {

        given : 'We create a tensor and immediately delete it.'
            Tensor t = Tensor.of(Shape.of(2), -3..2).getMut().delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to modify the data object...'
            t.mut.setDataAt(0, 7)

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to access the unsafe API of an already deleted tensor."

    }


    def 'A deleted tensor will throw an exception when accessing its data type.' () {

        given : 'We create a tensor and immediately delete it.'
            Tensor t = Tensor.of(Shape.of(2, 1), -3..2).getMut().delete()

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
            Tensor t = Tensor.of(Shape.of(2, 1), -3..2).getMut().delete()

        expect : 'This tensor should then know that it is deleted.'
            t.isDeleted()

        when : 'Trying to modify the data type...'
            t.mut.toType( Float.class )

        then : 'This should lead to a descriptive exception.'
            def exception = thrown(IllegalAccessError)
            exception.message == "Trying to access the unsafe API of an already deleted tensor."
    }


}
