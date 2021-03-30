package neureka.devices.file;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.AbstractBaseDevice;
import neureka.devices.Device;
import neureka.devices.file.heads.CSVHead;
import neureka.devices.file.heads.IDXHead;
import neureka.devices.file.heads.JPEGHead;

import java.io.File;
import java.io.IOException;
import java.util.*;
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
    private static final Map<String, FileDevice> _DEVICES = new WeakHashMap<>();

    private Map<Tsr<Number>, FileHead> _stored = new HashMap<>();

    @Getter private String _directory;
    @Getter private final List<String> _loadable = new ArrayList<>();
    @Getter private final List<String> _loaded = new ArrayList<>();

    public static FileDevice instance( String path ) {
        FileDevice device = _DEVICES.get( path );
        if ( device != null ) return device;
        device = new FileDevice( path );
        _DEVICES.put( path, device );
        return device;
    }

    private FileDevice( String directory ) {
        _directory = directory;
        _updateFolderView();
    }

    /**
     *  The underlying folder might change, files might be added or removed.
     *  In order to have an up-to-date view of the folder this method updates the current view state.
     */
    private void _updateFolderView() {
        File dir = new File( _directory );
        if ( ! dir.exists() ) dir.mkdirs();
        else {
            List<String> found = new ArrayList<>();
            File[] files = dir.listFiles();
            if ( files != null ) {
                for ( File file : files ) {
                    int i = file.getName().lastIndexOf( '.' );
                    if ( i > 0 ) {
                        String extension = file.getName().substring( i + 1 );
                        if ( FileHead.FACTORY.hasLoader( extension ) ) found.add( file.getName() );
                    }
                }
                _loadable.addAll( found ); // TODO!
            }
        }
    }

    public Tsr<?> load( String filename ) throws IOException {
        return load( filename, null );
    }

    public Tsr<?> load( String filename, Map<String, Object> conf ) throws IOException {
        if ( _loadable.contains( filename ) ) {
            String extension = filename.substring( filename.lastIndexOf( '.' ) + 1 );
            FileHead<?,?> head = FileHead.FACTORY.getLoader( extension ).load( _directory + "/" + filename, conf );
            assert head != null;
            Tsr tensor = head.load();
            _stored.put( tensor, head );
            _loadable.remove( filename );
            _loaded.add( filename );
            return tensor;
        }
        return null;
    }

    public FileHead<?, ?> fileHeadOf( Tsr<?> tensor ) {
        return _stored.get( tensor );
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
        int i = filename.lastIndexOf( '.' );
        if ( i < 1 ) {
            filename = filename + ".idx";
            i = filename.lastIndexOf( '.' );
        }
        String extension = filename.substring( i + 1 );
        if ( FileHead.FACTORY.hasSaver( extension ) ) {
            _stored.put(
                    tensor,
                    FileHead.FACTORY.getSaver(extension).save( _directory + "/" + filename, tensor, configurations )
            );
            tensor.setIsOutsourced(true);
        }
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
        throw new IllegalAccessError("FileDevice instances do not support executions.");
    }

    @Override
    public Object valueFor( Tsr<Number> tensor ) {
        return tensor.getValue();
    }

    @Override
    public Number valueFor( Tsr<Number> tensor, int index ) {
        return tensor.getValueAt( index );
    }

    @Override
    public Collection<Tsr<Number>> getTensors() {
        return _stored.keySet();
    }

    @Override
    public void update( Tsr<Number> oldOwner, Tsr<Number> newOwner ) {
        if ( _stored.containsKey( oldOwner ) ) {
            FileHead head = _stored.get( oldOwner );
            _stored.remove( oldOwner );
            _stored.put( newOwner, head );
        }
    }


}
