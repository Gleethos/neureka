package neureka.devices.file;

import neureka.Tensor;
import org.slf4j.LoggerFactory;

class PNGHandle extends AbstractImageFileHandle<PNGHandle>
{
    static {
        _LOG = LoggerFactory.getLogger( PNGHandle.class );
    }

    PNGHandle( String fileName ) { this(null, fileName); }

    PNGHandle(Tensor<Number> tensor, String filename ) {
        super(
                tensor,
                filename,
                new ImageFileType() {
                        @Override public Tensor.ImageType imageType()        { return Tensor.ImageType.ABGR_4BYTE; }
                        @Override public String        imageTypeName()    { return "png";                    }
                        @Override public String        defaultExtension() { return "png";                    }
                    }
            );
    }
}
