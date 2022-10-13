package neureka.devices.file;

import neureka.Tsr;
import org.slf4j.LoggerFactory;

/**
 *  This class is one of many extensions of the {@link AbstractFileHandle} which
 *  is therefore ultimately an implementation of the {@link FileHandle} interface.
 *  Like other {@link FileHandle} implementations of this class represents a file
 *  of a given type, in this case it represents a JPEG file.
 */
final class JPEGHandle extends AbstractImageFileHandle<JPEGHandle>
{
    static {
        _LOG = LoggerFactory.getLogger( JPEGHandle.class );
    }

    JPEGHandle( String fileName) { this( null, fileName ); }

    JPEGHandle( Tsr<Number> tensor, String filename ) {
        super(
                tensor,
                filename,
                new ImageFileType() {
                    @Override public Tsr.ImageType imageType() { return Tsr.ImageType.BGR_3BYTE; }

                    @Override public String imageTypeName() { return "jpeg"; }

                    @Override public String defaultExtension() { return "jpg"; }
                }
            );
    }

}
