package ut.tensors

import neureka.Tsr
import neureka.Tsr.ImageType
import spock.lang.Specification


class Tensor_Interop_Spec extends Specification
{

    def 'Not all tensor can be converted to images.'(
            Class<?> type, ImageType image, int... shape
    ) {

        when :
            Tsr.of(type).withShape(shape).all(-3).asImage(image)

        then :
            var exception = thrown(IllegalArgumentException)
        and :
            exception.message.length() > 10

        where :
            type | image                 | shape
            Byte | ImageType.BGR_3BYTE   | [3, 5]
            //Byte | ImageType.ARGB_1INT   | [7, 5, 3]
    }

}