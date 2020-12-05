package neureka.devices.storage;

import neureka.Tsr;
import neureka.calculus.backend.ExecutionCall;
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
public class FileDevice extends AbstractBaseDevice<Number>
{
    private static final Map<String, FileDevice> _DEVICES = new WeakHashMap<>();

    private static Map<String, Function<String, FileHead>> _LOADERS = Map.of(
            "idx", name -> {
                return new IDXHead( name );
            },
            "jpg", name -> {
                return new JPEGHead( name );
            },
            "png", name -> {
                return null;
            }
    );
    private static Map<String, BiFunction<String, Tsr<Number>, FileHead>> _SAVERS = Map.of(
            "idx", (name, tensor) -> {
                return new IDXHead( tensor, name );
            },
            "jpg", (name, tensor) -> {
                return new JPEGHead( tensor, name );
            },
            "png", (name, tensor) -> {
                return null;
            }
    );

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
    public Device restore(Tsr<Number> tensor ) {
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

    public Device store( Tsr<Number> tensor, String filename )
    {
        if ( filename.endsWith(".jpg") || filename.endsWith(".jpeg") ) {
            try {
                FileHead head = new JPEGHead( tensor, _directory + "/" + filename );
                _stored.put( tensor, head );
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            try {
                if ( !filename.endsWith(".idx") ) filename += ".idx";
                FileHead head = new IDXHead( tensor, _directory + "/" + filename );
                _stored.put( tensor, head );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
        }
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
