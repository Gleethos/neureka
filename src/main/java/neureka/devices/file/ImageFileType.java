package neureka.devices.file;

import neureka.Tensor;
import neureka.dtype.NumericType;

import java.util.Objects;

interface ImageFileType extends FileType
{
    default int numberOfChannels() { return this.imageType().numberOfChannels; }

    default NumericType<?,?,?,?> numericTypeRepresentation() {
        return ( (NumericType<?,?,?,?>) Objects.requireNonNull( imageType().dataType.getTypeClassInstance(NumericType.class) ) );
    }

    default Class<?> targetedValueType() {
        return this.numericTypeRepresentation().targetType();
    }

    Tensor.ImageType imageType();

    String imageTypeName();
}
