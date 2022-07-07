package neureka.devices.file.handles;

import neureka.Tsr;
import neureka.common.utility.LogUtil;
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
    private final ImageFileType _type;
    private int _width;
    private int _height;


    protected AbstractImageFileHandle( Tsr<Number> t, String filename, ImageFileType type ) {
        super( filename, type );
        LogUtil.nullArgCheck( type, "type", ImageFileType.class );
        _type = type;
        if ( t == null ) _loadHead();
        else
        {
            if ( t.rank() != 3 || t.rank() == 2 )
                throw new IllegalArgumentException(
                    "Expected tensor of rank 3, or 2 but encountered rank " + t.rank() + ". " +
                    "Cannot interpret tensor as image!"
                );

            if ( t.shape(t.rank()-1) != _type.numberOfChannels() )
                throw new IllegalArgumentException(
                    "Expected last tensor axes length " + t.shape(t.rank()-1) + " to be equal " +
                    "to " + _type.numberOfChannels() + ", the number of expected color channels!"
                );

            _height = t.shape(0);
            _width  = t.shape(1);
            t.setIsVirtual(false);
            store(t);
        }
    }


    private void _loadHead()
    {
        final File found = _loadFile();
        final BufferedImage image;

        try {
            image = ImageIO.read(found);
            Raster data = image.getData();
            _height = data.getHeight();
            _width = data.getWidth();
        } catch ( Exception exception ) {
            String message = _type.imageTypeName().toUpperCase() + " '"+_fileName+"' could not be read from file!";
            _LOG.error( message, exception );
            throw new IllegalStateException( message );
        }

        if ( _height < 1 || _width < 1 ) {
            String message = "The height and width of the " + _type + " at '"+_fileName+"' is "+_height+" & "+_width+"." +
                             "However both dimensions must at least be of size 1!";
            IllegalStateException e = new IllegalStateException( message );
            _LOG.error( message, e );
            throw e;
        }
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<Number> load() throws IOException {
        Object value = _loadData(); // This is simply some kind of primitive array.
        Tsr<?> t = Tsr.of(
                        _type.targetedValueType(),
                        new int[]{_height, _width, _type.numberOfChannels()},
                        value
                    );

        return t.getUnsafe().upcast(Number.class);
    }

    @Override protected Object _loadData() throws IOException
    {
        File found = _loadFile();
        BufferedImage image;
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
                        (start, end) -> { for (int i = start; i < end; i++) newData[i] = ui8.toTarget(data[i]); }
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

    /** {@inheritDoc} */
    @Override public int getValueSize() { return _width * _height * _type.numberOfChannels(); }

    /** {@inheritDoc} */
    @Override public int getDataSize() { return _width * _height * _type.numberOfChannels(); }

    /** {@inheritDoc} */
    @Override public int getTotalSize() { return _width * _height * _type.numberOfChannels(); }

    /** {@inheritDoc} */
    @Override public DataType<?> getDataType() { return DataType.of( UI8.class ); }

    /** {@inheritDoc} */
    @Override public int[] getShape() { return new int[]{ _height, _width, _type.numberOfChannels() }; }

    /** {@inheritDoc} */
    @Override
    public <T extends Number> Storage<Number> store( Tsr<T> tensor )
    {
        LogUtil.nullArgCheck( tensor, "tensor", Tsr.class );

        if ( _width != tensor.shape(1) )
            throw new IllegalArgumentException(
                "Cannot store tensor, because length " + tensor.shape(1) + " " +
                "of axis 1 is not equal to image width " + _width + "."
            );

        if ( _height != tensor.shape(0) )
            throw new IllegalArgumentException(
                    "Cannot store tensor, because length " + tensor.shape(0) + " " +
                            "of axis 0 is not equal to image width " + _height + "."
            );


        BufferedImage buff = tensor.asImage( _type.imageType() );

        try {
            ImageIO.write( buff, extension(), new File( _fileName ) );
        } catch ( Exception e ) {
            String message = "Failed writing tensor as " + extension() + " file!";
            _LOG.error(message, e);
            throw new IllegalStateException(message);
        }
        tensor.setIsOutsourced( true );
        tensor.getUnsafe().setDataType( DataType.of( _type.targetedValueType() ) );
        return this;
    }

}
