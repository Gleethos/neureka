package ut.dtype

import neureka.dtype.DataType
import neureka.dtype.NumericType
import neureka.dtype.custom.I16
import neureka.dtype.custom.UI8
import spock.lang.Specification

class DataType_Unit_Tests extends Specification
{

    def setupSpec()
    {
        reportHeader """
            This specification tests the "DataType" class.
        """
    }

    def 'DataType multiton instances behaive as expected.'(
      Class<?> typeClass, boolean isNumericType
    ) {
        given : 'A "DataType" instance representing / wrapping the relevant datatype Class passed to "instance(...)."'
            DataType dt = DataType.instance(typeClass)

        expect : 'The found instance is not null!'
            dt != null
        and : 'It contains the Class that it represents.'
            dt.getTypeClass() == typeClass
        and : 'This class either does or does not implement the "NumericType" interface.'
            dt.typeClassImplements(NumericType.class) == isNumericType

        where : 'The following data is being used :'
            typeClass    || isNumericType
            I16.class    || true
            byte[].class || false
            UI8.class    || true
            Float.class  || false
    }

}
