package neureka.device.storage;

import neureka.Component;
import neureka.Tsr;
import neureka.calculus.backend.ExecutionCall;
import neureka.device.Device;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

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
 */
public class FileDevice implements Device<Number>, Component<Tsr<Number>>
{
    private static Map<String, Function<String, FileHead>> _LOADERS = Map.of(
            "idx", name -> {
                return new IDXHead(name);
            },
            "jpg", name -> {
                return null;
            },
            "png", name -> {
                return null;
            }
    );

    private String _directory;
    private Map<Tsr<Number>, FileHead> _stored = new HashMap<>();

    public FileDevice( String directory ) {
        _directory = directory;
    }



    @Override
    public void dispose() {
            _stored = null;
            _directory = null;
    }

    @Override
    public Device get( Tsr<Number> tensor ) {
        return null;
    }

    @Override
    public Device add( Tsr<Number> tensor ) {
        return null;
    }

    @Override
    public Device add( Tsr<Number> tensor, Tsr<Number> parent ) {
        return null;
    }

    @Override
    public boolean has( Tsr<Number> tensor ) {
        return false;
    }

    @Override
    public Device rmv( Tsr<Number> tensor ) {
        return null;
    }

    @Override
    public Device cleaning(Tsr<Number> tensor, Runnable action) {
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
    public Device execute(ExecutionCall call) {
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
    public Collection<Tsr<Number>> tensors() {
        return null;
    }

    @Override
    public void update( Tsr<Number> oldOwner, Tsr<Number> newOwner ) {

    }
}
