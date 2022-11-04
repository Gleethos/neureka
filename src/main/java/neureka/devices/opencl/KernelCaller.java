package neureka.devices.opencl;

import neureka.Tsr;
import org.jocl.*;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

/**
 *  Instances of this class are utility factories provided by {@link OpenCLDevice} instances.
 *  When building new operations for tensors then this {@link KernelCaller} class is essential
 *  for calling compiled kernels residing within the gpu.
 */
public class KernelCaller
{
    private final cl_command_queue  _queue;
    private final cl_kernel         _kernel;
    private final List<Tsr<Number>> _inputs;

    private int _argId = 0;

    /**
     *
     * @param kernel The kernel which ought to be called.
     * @param queue The queue on which calls ought to be dispatched.
     */
    public KernelCaller( cl_kernel kernel, cl_command_queue queue ) {
        _queue  = queue;
        _kernel = kernel;
        _inputs = new ArrayList<>();
    }

    /**
     * This method passes 2 arguments to the kernel.
     * One for the data of the tensor and one for the configuration data!
     * @param tensor The tensor whose data and configuration ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory pattern).
     */
    public KernelCaller passAllOf( Tsr<Number> tensor ) {
        _inputs.add( tensor );
        clSetKernelArg( _kernel, _argId, Sizeof.cl_mem, Pointer.to( tensor.getMut().getData().getRef( OpenCLDevice.cl_tsr.class ).value.data ) );
        _argId++;
        return passConfOf( tensor );
    }

    /**
     *  This method passes the ND-Configuration in the form of a flattened int array to the kernel.
     *  Kernels can use this information for more complex indexing mechanisms as one would
     *  expect them to be present in tensor which have been reshaped or are simply
     *  slices of other tensors.
     *
     *  @param tensor The tensor whose ND configuration ought to be passed to the kernel.
     *  @return This very KernelCaller instance (factory pattern).
     */
    public KernelCaller passConfOf( Tsr<Number> tensor ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_mem, Pointer.to( tensor.getMut().getData().getRef( OpenCLDevice.cl_tsr.class ).config.data ) );
        _argId++;
        return this;
    }

    /**
     * This method passes 1 argument to the kernel.
     * Namely, the data of the tensor!
     * @param tensor The tensor whose data ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory pattern).
     */
    public <T extends Number> KernelCaller pass( Tsr<T> tensor ) {
        _inputs.add( tensor.getMut().upcast(Number.class) );
        clSetKernelArg( _kernel, _argId, Sizeof.cl_mem, Pointer.to( tensor.getMut().getData().getRef( OpenCLDevice.cl_tsr.class ).value.data ) );
        _argId++;
        return this;
    }

    /**
     *
     * @param value An int value which ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory pattern).
     */
    public KernelCaller pass( int value ) {
        return this.pass( new int[]{ value } );
    }

    /**
     *  Use this to pass an array of int values to the kernel.
     *
     * @param values An array of int values which ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory pattern).
     */
    public KernelCaller pass( int... values ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_int * (long) values.length, Pointer.to( values ) );
        _argId++;
        return this;
    }

    /**
     *  Use this to pass an array of float values to the kernel.
     *
     * @param values An array of float values which ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory pattern).
     */
    public KernelCaller pass( float... values ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_float * (long) values.length, Pointer.to( values ) );
        _argId++;
        return this;
    }

    public KernelCaller pass( double... values ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_double * (long) values.length, Pointer.to( values ) );
        _argId++;
        return this;
    }

    public KernelCaller pass( short... values ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_short * (long) values.length, Pointer.to( values ) );
        _argId++;
        return this;
    }

    public KernelCaller pass( long... values ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_long * (long) values.length, Pointer.to( values ) );
        _argId++;
        return this;
    }

    public KernelCaller pass( byte... values ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_char * (long) values.length, Pointer.to( values ) );
        _argId++;
        return this;
    }

    /**
     * @param value A float value which ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory pattern).
     */
    public KernelCaller pass( float value ) {
        return this.pass( new float[]{ value } );
    }

    public KernelCaller pass( double value ) {
        return this.pass( new double[]{ value } );
    }

    public KernelCaller pass( short value ) {
        return this.pass( new short[]{ value } );
    }

    public KernelCaller pass( long value ) {
        return this.pass( new long[]{ value } );
    }

    public KernelCaller pass( byte value ) {
        return this.pass( new byte[]{ value } );
    }

    public KernelCaller pass( Number value ) {
        if ( value instanceof Float ) return this.pass( value.floatValue() );
        else if ( value instanceof Double ) return this.pass( value.doubleValue() );
        else if ( value instanceof Integer ) return this.pass( value.intValue() );
        else if ( value instanceof Long ) return this.pass( value.longValue() );
        else if ( value instanceof Short ) return this.pass( value.shortValue() );
        else if ( value instanceof Byte ) return this.pass( value.byteValue() );
        else throw new IllegalArgumentException( "Unsupported number type: " + value.getClass().getName() );
    }

    public KernelCaller passLocalFloats( long size ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_float * (long) size, null );
        _argId++;
        return this;
    }

    /**
     *
     * @param globalWorkSize The number of global threads which will be dispatched.
     */
    public void call( int globalWorkSize )
    {
        cl_event[] events = _getWaitList( _inputs.toArray( new Tsr[ 0 ] ) );
        if ( events.length > 0 ) {
            clWaitForEvents( events.length, events );
            _releaseEvents( _inputs.toArray( new Tsr[ 0 ] ) );
        }
        clEnqueueNDRangeKernel(
                _queue, _kernel,
                1,
                null,
                new long[]{ globalWorkSize },
                null,
                0,
                null,
                null
        );
    }

     /**
     *  Use this to call the kernel with 2 long arrays defining how the kernel should be indexed and parallelized.
     *  The {@code globalWorkSizes} span an n-dimensional grid of global threads,
     *  whereas the {@code localWorkSizes} defines the dimensions of a grid of local work items (which are called "work groups").
     *  The total number of work items is equal to the product of the {@code localWorkSizes} array entries
     *  exactly like the product of the {@code globalWorkSizes} array is the total number of (global) threads.        <br>
     *  Both sizes have to fulfill the following condition: {@code globalWorkSize = localWorkSize * numberOfGroups}. <br>
     *  Note: The {@code localWorkSizes} is optional, so the second argument may be null
     *  in which case OpenCL will choose a local group size appropriately for you.
     *  This is usually also the optimal choice,
     *  however if the global work size is a prime number (that is larger than the maximum local work size),
     *  then an OpenCL implementation may be forced to use a local work size of 1...
     * <p>
     * This can usually be circumvented by padding the data to be a multiple of a more appropriate
     * local work size or by introducing boundary checks in your kernel.
     *
     * @param globalWorkSizes An arrays of long values which span a nd-grid of global threads.
     * @param localWorkSizes  An arrays of long values which span a nd-grid of local threads (work groups).
     */
    public void call( long[] globalWorkSizes, long[] localWorkSizes )
    {
        cl_event[] events = _getWaitList( _inputs.toArray( new Tsr[ 0 ] ) );
        if ( events.length > 0 ) {
            clWaitForEvents( events.length, events );
            _releaseEvents( _inputs.toArray( new Tsr[ 0 ] ) );
        }
        assert localWorkSizes == null || globalWorkSizes.length == localWorkSizes.length;
        clEnqueueNDRangeKernel(
                _queue, _kernel,
                globalWorkSizes.length,
                null,
                globalWorkSizes,
                localWorkSizes,
                0,
                null,
                null
        );
    }

    
    private void _releaseEvents( Tsr<Number>[] tensors ) {
        for ( Tsr<Number> t : tensors ) {
            if ( t.getMut().getData().getRef( OpenCLDevice.cl_tsr.class ).value.event != null ) {
                clReleaseEvent(t.getMut().getData().getRef( OpenCLDevice.cl_tsr.class ).value.event);
                t.getMut().getData().getRef( OpenCLDevice.cl_tsr.class ).value.event = null;
            }
        }
    }

    
    private cl_event[] _getWaitList( Tsr<Number>[] tensors ) {
        List<cl_event> list = new ArrayList<>();
        for ( Tsr<Number> t : tensors ) {
            cl_event event = t.getMut().getData().getRef( OpenCLDevice.cl_tsr.class ).value.event;
            if ( event != null && !list.contains(event) ) {
                list.add( event );
            }
        }
        return list.toArray( new cl_event[ 0 ] );
    }

}
