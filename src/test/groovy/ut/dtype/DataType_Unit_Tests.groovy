package ut.dtype

import neureka.dtype.DataType
import neureka.dtype.NumericType
import neureka.dtype.custom.F32
import neureka.dtype.custom.I16
import neureka.dtype.custom.I8
import neureka.dtype.custom.UI8
import spock.lang.Specification

class DataType_Unit_Tests extends Specification
{

    def setupSpec()
    {
        reportHeader """
            This specification tests the "DataType" class, which hosts a multiton
            design pattern in order to guarantee uniqueness of instances of the type
            which represent the same class type. <br>
            Instances of this class wrap a Class variable which is the type of the data of the tensor. <br>
            (The following types are usually used : UI8, I8, UI16, I16, UI32, I64, I32, F32, F64 )
        """
    }

    def 'DataType multiton instances behave as expected.'(
      Class<?> typeClass, Class<?> targetClass, boolean isNumericType
    ) {
        given : 'A "DataType" instance representing / wrapping the relevant datatype Class passed to "instance(...)."'
            DataType dt = DataType.instance( typeClass )

        expect : 'The found instance is not null!'
            dt != null
        and : 'It contains the Class that it represents.'
            dt.getTypeClass() == targetClass
        and : 'This class either does or does not implement the "NumericType" interface.'
            dt.typeClassImplements(NumericType.class) == isNumericType

        where : 'The following data is being used :'
            typeClass           ||  targetClass        | isNumericType
            I16.class           ||   I16.class         | true
            byte[].class        ||   I8.class          | true
            UI8.class           ||   UI8.class         | true
            Float.class         ||   F32.class         | true
            String.class        ||   String.class      | false
            Object.class        ||   Object.class      | false
            Specification.class || Specification.class | false
    }



}
