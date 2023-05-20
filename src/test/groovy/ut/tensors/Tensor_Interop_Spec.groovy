package ut.tensors


import neureka.Tensor
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Tensors play well with other data structures!")
@Narrative('''

    Tensors should have good interoperability with other JDK data structures like images.
    In this specification we define these interoperability requirements.

''')
class Tensor_Interop_Spec extends Specification
{

    def 'Tensor can be converted to buffered images.'(
            Class<?> type, Tensor.ImageType image, int... shape
    ) {
        when : 'A tensor of the provided type is converted to a buffered image'
            var asImage = Tensor.of(type).withShape(shape).andFill(42..73).asImage(image)

        then : 'The resulting image has the expected shape'
            asImage.height == shape[0]
            asImage.width  == shape[1]

        where :
            type    | image                        | shape
            Byte    | Tensor.ImageType.BGR_3BYTE      | [3, 5, 3]
            Integer | Tensor.ImageType.ARGB_1INT      | [7, 5, 1]
            Byte    | Tensor.ImageType.ABGR_4BYTE     | [7, 5, 4]
            Byte    | Tensor.ImageType.ABGR_PRE_4BYTE | [7, 5, 4]
    }


    def 'Not all tensor can be converted to images.'(
            Class<?> type, Tensor.ImageType image, int... shape
    ) {

        when : 'We try to convert a tensor of the provided type to a buffered image...'
            Tensor.of(type).withShape(shape).all(-3).asImage(image)

        then : 'An exception is thrown!'
            var exception = thrown(IllegalArgumentException)
        and : 'The exception message has a plausible size.'
            exception.message.length() > 13

        where :
            type    | image                     | shape
            Byte    | Tensor.ImageType.BGR_3BYTE  | [3, 5]
            Integer | Tensor.ImageType.ARGB_1INT  | [7, 5, 3]
            String  | Tensor.ImageType.ARGB_1INT  | [7, 5, 1]
            Byte    | Tensor.ImageType.ABGR_4BYTE | [2, 9, 3]
    }

}