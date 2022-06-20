package neureka.devices.file.handles;

import neureka.Tsr;

public class PNGHandle extends AbstractImageFileHandle<PNGHandle>
{
    public PNGHandle( String fileName ) { this(null, fileName); }

    public PNGHandle( Tsr<Number> tensor, String filename ) {
        super(
                tensor,
                filename,
                new ImageFileType() {
                        @Override public Tsr.ImageType imageType()        { return Tsr.ImageType.ABGR_4BYTE; }
                        @Override public String        imageTypeName()    { return "png";                    }
                        @Override public String        defaultExtension() { return "png";                    }
                    }
            );
    }
}
