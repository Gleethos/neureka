package neureka.devices.storage;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.AbstractBaseDevice;
import neureka.devices.Device;

import java.io.File;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 *  This Device implementation is responsible for reading and or writing
 *  tensors to a given directory.
 *  The directory which ought to be governed by an instance of this
 *  class has to be passed to the constructor (as relative path) after which the device
 *  reads the files within this directory making the tensors accessible.
 *  Tensors on a file device however are not loaded onto memory entirely, instead
 *  a mere file handle for each "file tensor" is being instantiated.
 *  Therefore tensors that are stored on this device are not fit for computation.
 *  The "get(..)" method has to be called instead.
 *
 *  The abstraction provided by the "Device" interface
 *  does not necessitate that concrete implementations
 *  represent accelerator hardware. <br>
 *  Generally speaking a device is a thing that
 *  stores tensors and optionally also handles "ExecutionCall" instances.
 *  Therefore, an implementation might also represent a simple
 *  storage device like your local SSD ord HDD...
 */
@Accessors( prefix = {"_"} )
@ToString
public class FileDevice extends AbstractBaseDevice<Number>
{
    interface Loader { FileHead load( String name, Map<String, Object> config ); }
    interface Saver { FileHead save( String name, Tsr tensor, Map<String, Object> config ); }

    private static final Map<String, FileDevice> _DEVICES = new WeakHashMap<>();

    private static Map<String, Loader> _LOADERS;
    static {
        _LOADERS = new HashMap<>();
        _LOADERS.put( "idx", ( name, conf ) -> new IDXHead( name ) );
        _LOADERS.put( "jpg", ( name, conf ) -> new JPEGHead( name ) );
        _LOADERS.put( "png", ( name, conf ) -> null ); // TODO!
        _LOADERS.put( "csv", ( name, conf ) -> new CSVHead( name, conf ) );
    }

    private static Map<String, Saver> _SAVERS;
    static {
        _SAVERS = new HashMap<>();
        _SAVERS.put( "idx", ( name, tensor, conf ) -> new IDXHead( tensor, name ) );
        _SAVERS.put( "jpg", ( name, tensor, conf ) -> new JPEGHead( tensor, name ) );
        _SAVERS.put( "png", ( name, tensor, conf ) -> null ); // TODO!
    }

    @Getter
    private String _directory;
    private Map<Tsr<Number>, FileHead> _stored = new HashMap<>();

    public static FileDevice instance( String path ) {
        FileDevice device = _DEVICES.get( path );
        if ( device != null ) return device;
        device = new FileDevice( path );
        _DEVICES.put( path, device );
        return device;
    }

    private FileDevice( String directory ) {
        _directory = directory;
        File dir = new File( directory );
        if ( ! dir.exists() ) {
            dir.mkdirs();
        }
    }

    public FileHead<?, Number> fileHeadOf( Tsr<Number> tensor ) {
        return _stored.get(tensor);
    }

    @Override
    public void dispose() {
            _stored = null;
            _directory = null;
    }

    @Override
    public Device restore( Tsr<Number> tensor ) {
        if ( !this.has( tensor ) )
            throw new IllegalStateException( "The given tensor is not stored on this file device." );
        FileHead head = _stored.get( tensor );
        try {
            head.restore( tensor );
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public Device store( Tsr<Number> tensor )
    {
        if ( this.has( tensor ) ) {
            FileHead head = _stored.get( tensor );
            try {
                head.store( tensor );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
            return this;
        }
        String filename = tensor.shape().stream().map( Object::toString ).collect(Collectors.joining("x"));
        filename = "tensor_" + filename + "_" + tensor.getDataType().getTypeClass().getSimpleName().toLowerCase();
        filename = filename + "_" + java.time.LocalDate.now().toString();
        filename = filename + "_" + java.time.LocalTime.now().toString();
        filename = filename.replace( ".", "_" ).replace( ":","-" ) + "_.idx";
        store( tensor, filename );
        return this;
    }

    public FileDevice store( Tsr<Number> tensor, String filename )
    {
        return store( tensor, filename, null );
    }

    public FileDevice store( Tsr<Number> tensor, String filename, Map<String, Object> configurations )
    {
        FileHead head = null;
        if ( filename.endsWith(".jpg") || filename.endsWith(".jpeg") ) {
            try {
                head = new JPEGHead( tensor, _directory + "/" + filename );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
        } else if ( filename.endsWith(".csv") ) {
            try {
                head = new CSVHead( tensor, _directory + "/" + filename );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
        } else {
            try {
                if ( !filename.endsWith(".idx") ) filename += ".idx";
                head = new IDXHead( tensor, _directory + "/" + filename );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
        }
        _stored.put( tensor, head );
        tensor.setIsOutsourced( true );
        return this;
    }

    @Override
    public Device store( Tsr<Number> tensor, Tsr<Number> parent ) {
        return null;
    }

    @Override
    public boolean has( Tsr<Number> tensor ) {
        return _stored.containsKey( tensor );
    }

    @Override
    public Device free( Tsr<Number> tensor )
    {
        if ( !this.has( tensor ) )
            throw new IllegalStateException( "The given tensor is not stored on this file device." );
        FileHead head = _stored.get( tensor );
        try {
            head.free();
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        _stored.remove( tensor );
        return this;
    }

    @Override
    public Device cleaning( Tsr<Number> tensor, Runnable action ) {
        return this;
    }

    @Override
    public Device overwrite64( Tsr<Number> tensor, double[] value ) {
        return null;
    }

    @Override
    public Device overwrite32( Tsr<Number> tensor, float[] value ) {
        return null;
    }

    @Override
    public Device swap( Tsr<Number> former, Tsr<Number> replacement ) {
        return null;
    }

    @Override
    public Device execute( ExecutionCall call ) {
        return null;
    }

    @Override
    public double[] value64f( Tsr<Number> tensor ) {
        return new double[0];
    }

    @Override
    public float[] value32f( Tsr<Number> tensor ) {
        return new float[0];
    }

    @Override
    public double value64f( Tsr<Number> tensor, int index ) {
        return 0;
    }

    @Override
    public float value32f( Tsr<Number> tensor, int index ) {
        return 0;
    }

    @Override
    public Collection<Tsr<Number>> getTensors() {
        return _stored.keySet();
    }

    @Override
    public void update( Tsr<Number> oldOwner, Tsr<Number> newOwner ) {

    }


}
