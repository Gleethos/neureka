package ut.tensors


import neureka.Tensor
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Tensor Type Conversion")
@Narrative('''

    Here we specify how a tensor can be converted to other data types
    like for example another tensor of a different data type.

''')
@Subject([Tensor])
class Tensor_Conversion_Spec extends Specification
{

    def 'We turn a tensor into a scalar value or string through the "as" operator!'()
    {
        given : 'A tensor of 3 floats:'
            var t = Tensor.ofFloats().vector(42, 42, 42)

        expect : 'We can now turn the tensor int other data types!'
            (t as Integer) == 42
            (t as Double) == 42
            (t as Short) == 42
            (t as Byte) == 42
            (t as Long) == 42
        and : 'Also use it instead of the "toString" method.'
            (t as String) == t.toString()
    }

    def 'Tensors value type can be changed by calling "toType(...)".'()
    {
        reportInfo("""
            Warning! The `toType` method mutates the tensor!
            This is especially problematic with respect to generics, 
            because if the tensor is still used as a tensor of the old type, 
            then the compiler will not be able to detect that the tensor has changed its type.
            This is why we have to use the `unsafe` API exists.
            Only use this method if there are urgent performance requirements and
            you know exactly what you are doing!
        """)

        given :
            Tensor x = Tensor.of(3d)

        when : x.mut.toType( Float.class )
        then :
            x.rawItems instanceof float[]
            x.mut.data.get() instanceof float[]
            x.rawData instanceof float[]
            x.getItemsAs( float[].class )[ 0 ] == 3.0f

        when : x.mut.toType( Double.class )
        then :
            x.rawItems instanceof double[]
            x.mut.data.get() instanceof double[]
            x.rawData instanceof double[]
            x.getItemsAs( float[].class )[ 0 ]==3.0f
    }

    def 'We can change the data type of all kinds of tensors.'(
            Class<?> sourceType, Class<?> targetType
    ) {
        reportInfo("""
            Warning! The `toType` method mutates the tensor!
            This is especially problematic with respect to generics, 
            because if the tensor is still used as a tensor of the old type, 
            then the compiler will not be able to detect that the tensor has changed its type.
            This is why we have to use the `unsafe` API exists.
            Only use this method if there are urgent performance requirements and
            you know exactly what you are doing!
        """)

        given : 'A simple tensor with a few initial values.'
            var data = [-3, -12, 42, -42, 12, 3]
            var a = Tensor.of(sourceType).withShape(data.size()).andFill(data)

        when : 'We change the data type of the tensor using the unsafe "toType" method.'
            var b = a.mut.toType(targetType)

        then : 'The returned tensor has the expected data type.'
            b.itemType == targetType
        and : 'The returned tensor has the expected values.'
            b.rawItems == data.collect({ it.asType(targetType) })
        and : 'The returned tensor is in fact the original instance.'
            a === b

        where : 'We use the following data and matrix dimensions!'
            sourceType | targetType
            Double     | Double
            Double     | Float
            Double     | Long
            Double     | Integer
            Double     | Short
            Double     | Byte
            Float      | Double
            Float      | Float
            Float      | Long
            Float      | Integer
            Float      | Short
            Float      | Byte
            Long       | Double
            Long       | Float
            Long       | Long
            Long       | Integer
            Long       | Short
            Long       | Byte
            Integer    | Double
            Integer    | Float
            Integer    | Long
            Integer    | Integer
            Integer    | Short
            Integer    | Byte
            Short      | Double
            Short      | Float
            Short      | Long
            Short      | Integer
            Short      | Short
            Short      | Byte
            Byte       | Double
            Byte       | Float
            Byte       | Long
            Byte       | Integer
            Byte       | Short
            Byte       | Byte
    }



}
