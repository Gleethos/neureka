package neureka.devices.file.handles;

import neureka.Tsr;
import neureka.devices.Storage;
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

public abstract class AbstractImageFileHandle<C> extends AbstractFileHandle<C, Number>
{
    static {
        _LOG = LoggerFactory.getLogger( AbstractImageFileHandle.class );
    }

    private final String _type;
    private int _width;
    private int _height;


    public AbstractImageFileHandle( String type, String fileName )
    {
        super( fileName );
        try {
            _loadHead();
        } catch( Exception e ) {
            _LOG.error("Failed reading JPG file!");
        }
        _type = type;
    }

    protected AbstractImageFileHandle( String type, Tsr<Number> t, String filename ) {
        super( filename );
        assert t.rank() == 3;
        assert t.shape( 2 ) == 3;
        _type = type;
        _height = t.shape(0);
        _width = t.shape(1);
        t.setIsVirtual( false );
        store( t );
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
            String message = _type.toUpperCase() + " '"+_fileName+"' could not be read from file!";
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
        Tsr t = Tsr.of( I16.class, new int[]{_height, _width, 3}, value );
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
            short[] newData = new short[ data.length ];

            UI8 ui8 = new UI8();
            CPU.get().getExecutor().threaded(
                    data.length,
                    ( start, end ) -> {
                        for ( int i=start; i<end; i++ ) newData[i] = ui8.toTarget( data[i] );
                    }
            );
            return newData;
        }
        catch ( IOException e )
        {
            _LOG.error( "Failed loading " + _type + " file!", e );
            throw e;
        }
    }

    @Override
    public int getValueSize() {
        return _width * _height * 3;
    }

    @Override
    public int getDataSize() {
        return _width * _height * 3;
    }

    @Override
    public int getTotalSize() {
        return _width * _height * 3;
    }

    @Override
    public DataType<?> getDataType() {
        return DataType.of( UI8.class );
    }

    @Override
    public int[] getShape() {
        return new int[]{ _height, _width, 3 };
    }

    @Override
    public <T extends Number> Storage<Number> store(Tsr<T> tensor )
    {
        assert tensor.shape(1) == _width;
        assert tensor.shape(0) == _height;

        BufferedImage buffi = tensor.asImage(Tsr.ImageType.BGR_3BYTE);

        try {
            ImageIO.write( buffi, extension(), new File( _fileName ) );
        } catch ( Exception e ) {
            _LOG.error("Failed writing tensor to " + extension() + "!", e);
            return this;
        }
        tensor.setIsOutsourced( true );
        tensor.getUnsafe().setDataType( DataType.of( I16.class ) );
        return this;
    }

}
