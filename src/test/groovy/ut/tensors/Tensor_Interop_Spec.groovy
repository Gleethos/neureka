package ut.tensors

import neureka.Tsr
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
            Class<?> type, Tsr.ImageType image, int... shape
    ) {
        when :
            var asImage = Tsr.of(type).withShape(shape).andFill(42..73).asImage(image)

        then :
            asImage.height == shape[0]
            asImage.width  == shape[1]

        where :
            type    | image                        | shape
            Byte    | Tsr.ImageType.BGR_3BYTE      | [3, 5, 3]
            Integer | Tsr.ImageType.ARGB_1INT      | [7, 5, 1]
            Byte    | Tsr.ImageType.ABGR_4BYTE     | [7, 5, 4]
            Byte    | Tsr.ImageType.ABGR_PRE_4BYTE | [7, 5, 4]
    }


    def 'Not all tensor can be converted to images.'(
            Class<?> type, Tsr.ImageType image, int... shape
    ) {

        when :
            Tsr.of(type).withShape(shape).all(-3).asImage(image)

        then :
            var exception = thrown(IllegalArgumentException)
        and :
            exception.message.length() > 13

        where :
            type    | image                     | shape
            Byte    | Tsr.ImageType.BGR_3BYTE   | [3, 5]
            Integer | Tsr.ImageType.ARGB_1INT   | [7, 5, 3]
            String  | Tsr.ImageType.ARGB_1INT   | [7, 5, 1]
            Byte    | Tsr.ImageType.ABGR_4BYTE  | [2, 9, 3]
    }

}