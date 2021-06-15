package neureka.devices.file.heads;

import neureka.Tsr;
import neureka.devices.Storage;
import neureka.devices.host.HostCPU;
import neureka.dtype.DataType;
import neureka.dtype.custom.I16;
import neureka.dtype.custom.UI8;
import neureka.utility.DataConverter;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.io.*;

/**
 *  This class is one of many extensions of the {@link AbstractFileHead} which
 *  is therefore ultimately an implementation of the {@link neureka.devices.file.FileHead} interface.
 *  Like other {@link neureka.devices.file.FileHead} implementations this class represents a file
 *  of a given type, in this case it represents a CSV file.
 */
public class JPEGHead extends AbstractFileHead<JPEGHead, Number>
{
    static {
        _LOG = LoggerFactory.getLogger( JPEGHead.class );
    }

    int _width;
    int _height;
    //int _totalSize;

    public JPEGHead( String fileName )
    {
        super( fileName );
        try {
            _loadHead();
        } catch( Exception e ) {
            System.err.print("Failed reading JPG file!");
        }
    }

    public JPEGHead( Tsr<Number> t, String filename ) {
        super( filename );
        assert t.rank() == 3;
        assert t.shape( 2 ) == 3;
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
            String message = "JPEG '"+_fileName+"' could not be read from file!";
            _LOG.error( message, exception );
            exception.printStackTrace();
        }

        try
        {
            if ( _height < 1 || _width < 1 ) {
                String message = "The height and width of the jpeg at '"+_fileName+"' is "+_height+" & "+_width+"." +
                        "However both dimensions must at least be of size 1!";
                Exception e = new IOException( message );
                _LOG.error( message, e );
                throw e;
            }
        }
        catch ( Exception e )
        {
            e.printStackTrace();
        }
    }

    @Override
    public Tsr<Number> load() throws IOException {
        Object value = _loadData();
        Tsr<Number> t = Tsr.of( new int[]{_height, _width, 3}, I16.class, value );
        return t;
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
            HostCPU.instance().getExecutor().threaded(
                    data.length,
                    ( start, end ) -> {
                        for ( int i=start; i<end; i++ ) newData[i] = ui8.toTarget( data[i] );
                    }
            );
            return newData;
        }
        catch ( IOException e )
        {
            e.printStackTrace();
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
    public String extension() {
        return "jpg";
    }

    @Override
    public Storage<Number> store( Tsr<Number> tensor )
    {
        byte[] data = DataConverter.instance().convert( tensor.getData(), byte[].class );

        BufferedImage buffi = new BufferedImage( _width, _height, BufferedImage.TYPE_3BYTE_BGR );
        buffi.setData(
                Raster.createRaster(
                        buffi.getSampleModel(), new DataBufferByte( data, data.length ),
                        new Point()
                )
        );
        try {
            ImageIO.write( buffi, "jpg", new File( _fileName ) );
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        tensor.setIsOutsourced( true );
        tensor.setDataType( DataType.of( I16.class ) );
        return this;
    }


}
