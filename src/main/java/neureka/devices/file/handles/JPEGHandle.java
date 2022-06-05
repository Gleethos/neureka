package neureka.devices.file.handles;

import neureka.Tsr;
import neureka.devices.Storage;
import neureka.devices.file.FileHandle;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.dtype.custom.I16;
import neureka.dtype.custom.UI8;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;

/**
 *  This class is one of many extensions of the {@link AbstractFileHandle} which
 *  is therefore ultimately an implementation of the {@link FileHandle} interface.
 *  Like other {@link FileHandle} implementations of this class represents a file
 *  of a given type, in this case it represents a JPEG file.
 */
public final class JPEGHandle extends AbstractImageFileHandle<JPEGHandle>
{

    public JPEGHandle( String fileName) { this( null, fileName ); }

    public JPEGHandle(Tsr<Number> t, String filename) {
        super(t, filename, new ImageFileType() {
            @Override public int numberOfChannels() { return 3; }

            @Override public Tsr.ImageType imageType() { return Tsr.ImageType.BGR_3BYTE; }

            @Override public String imageTypeName() { return "jpeg"; }

            @Override public String defaultExtension() { return "jpg"; }
        });
    }

}
