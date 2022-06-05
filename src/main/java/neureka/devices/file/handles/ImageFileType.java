package neureka.devices.file.handles;

import neureka.Tsr;

interface ImageFileType extends FileType
{
    int numberOfChannels();

    Tsr.ImageType imageType();

    String imageTypeName();
}
