package neureka.devices.file.handles;

import neureka.Tsr;
import neureka.devices.Storage;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.dtype.custom.UI8;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;

public abstract class AbstractImageFileHandle<C> extends AbstractFileHandle<C, Number>
{
    static {
        _LOG = LoggerFactory.getLogger( AbstractImageFileHandle.class );
    }

    private final ImageFileType _type;
    private int _width;
    private int _height;


    protected AbstractImageFileHandle( Tsr<Number> t, String filename, ImageFileType type ) {
        super( filename, type );
        _type = type;
        if ( t == null ) {
            try {
                _loadHead();
            } catch( Exception e ) {
                _LOG.error("Failed reading JPG file!");
            }
        } else {
            assert t.rank() == 3;
            assert t.shape(2) == _type.numberOfChannels();
            _height = t.shape(0);
            _width = t.shape(1);
            t.setIsVirtual(false);
            store(t);
        }
    }


    private void _loadHead()
    {
        File found = _loadFile();

        BufferedImage image = null;
        Raster data;

        try {
            image = ImageIO.read(found);
            data = image.getData();
            _height = data.getHeight();
            _width = data.getWidth();
        } catch ( Exception exception ) {
            String message = _type.imageTypeName().toUpperCase() + " '"+_fileName+"' could not be read from file!";
            _LOG.error( message, exception );
            exception.printStackTrace();
        }

        try
        {
            if ( _height < 1 || _width < 1 ) {
                String message = "The height and width of the " + _type + " at '"+_fileName+"' is "+_height+" & "+_width+"." +
                        "However both dimensions must at least be of size 1!";
                Exception e = new IOException( message );
                _LOG.error( message, e );
                throw e;
            }
        }
        catch ( Exception e )
        {
            _LOG.error( "Failed loading " + _type + " file!", e );
        }
    }

    @Override
    public Tsr<Number> load() throws IOException {
        Object value = _loadData();
        Tsr t = Tsr.of(
                    _type.targetedValueType(),
                    new int[]{_height, _width, _type.numberOfChannels()},
                    value
                );
        return (Tsr<Number>) t;
    }

    @Override
    protected Object _loadData() throws IOException
    {
        File found = _loadFile();
        BufferedImage image = null;
        try
        {
            image = ImageIO.read( found );
            byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            if ( data.length != (_height * _width * _type.numberOfChannels()) )
                throw new IllegalStateException("Loaded image data array does not match expected number of elements!");

            if ( _type.targetedValueType() == Short.class ) {
                short[] newData = new short[data.length];
                UI8 ui8 = new UI8();
                CPU.get().getExecutor().threaded(
                        data.length,
                        (start, end) -> {
                            for (int i = start; i < end; i++) newData[i] = ui8.toTarget(data[i]);
                        }
                );
                return newData;
            }
            else throw new IllegalStateException("Alternative types not yet supported!");
        }
        catch ( IOException e )
        {
            _LOG.error( "Failed loading " + _type + " file!", e );
            throw e;
        }
    }

    @Override
    public int getValueSize() {
        return _width * _height * _type.numberOfChannels();
    }

    @Override
    public int getDataSize() {
        return _width * _height * _type.numberOfChannels();
    }

    @Override
    public int getTotalSize() {
        return _width * _height * _type.numberOfChannels();
    }

    @Override
    public DataType<?> getDataType() {
        return DataType.of( UI8.class );
    }

    @Override
    public int[] getShape() {
        return new int[]{ _height, _width, _type.numberOfChannels() };
    }

    @Override
    public <T extends Number> Storage<Number> store( Tsr<T> tensor )
    {
        assert tensor.shape(1) == _width;
        assert tensor.shape(0) == _height;

        BufferedImage buffi = tensor.asImage( _type.imageType() );

        try {
            ImageIO.write( buffi, extension(), new File( _fileName ) );
        } catch ( Exception e ) {
            _LOG.error("Failed writing tensor as " + extension() + " file!", e);
            return this;
        }
        tensor.setIsOutsourced( true );
        tensor.getUnsafe().setDataType( DataType.of( _type.targetedValueType() ) );
        return this;
    }

}
