package neureka.devices.file.handles;

import neureka.Tsr;

public class PNGHandle extends AbstractImageFileHandle<PNGHandle>
{
    public PNGHandle( String fileName ) { this(null, fileName); }

    public PNGHandle( Tsr<Number> t, String filename ) {
        super(t, filename, new ImageFileType() {
            @Override public int numberOfChannels() { return 4; }

            @Override public Tsr.ImageType imageType() { return Tsr.ImageType.ABGR_4BYTE; }

            @Override public String imageTypeName() { return "png"; }

            @Override public String defaultExtension() { return "png"; }
        });
    }
}
