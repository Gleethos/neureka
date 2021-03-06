package neureka.devices.opencl;

import neureka.Tsr;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jocl.*;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

/**
 *  Instances of this class are utility factories provided by OpenCLDevice instances.
 *  When building new operations for tensors then this KernelBuilder class is essential
 *  for calling compiled kernels residing within the gpu.
 */
public class KernelCaller
{
    private final cl_command_queue _queue;
    private final cl_kernel _kernel;
    private final List<Tsr<Number>> _inputs;

    private int _argId = 0;

    /**
     *
     * @param kernel The kernel which ought to be called.
     * @param queue The queue on which calls ought to be dispatched.
     */
    public KernelCaller( @NotNull cl_kernel kernel, @NotNull cl_command_queue queue ) {
        _queue = queue;
        _kernel = kernel;
        _inputs = new ArrayList<>();
    }

    /**
     * This method passes 2 arguments to the kernel.
     * One for the data of the tensor and one for the configuration data!
     * @param tensor The tensor whose data and configuration ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory patter).
     */
    public KernelCaller pass( @NotNull Tsr<Number> tensor ) {
        _inputs.add( tensor );
        clSetKernelArg( _kernel, _argId, Sizeof.cl_mem, Pointer.to( tensor.find( OpenCLDevice.cl_tsr.class ).value.data ) );
        _argId++;
        clSetKernelArg( _kernel, _argId, Sizeof.cl_mem, Pointer.to( tensor.find( OpenCLDevice.cl_tsr.class ).config.data ) );
        _argId++;
        return this;
    }

    /**
     * This method passes 1 argument to the kernel.
     * Namely the data of the tensor!
     * @param tensor The tensor whose data ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory patter).
     */
    public KernelCaller passRaw( @NotNull Tsr<Number> tensor ) {
        _inputs.add( tensor );
        clSetKernelArg( _kernel, _argId, Sizeof.cl_mem, Pointer.to( tensor.find( OpenCLDevice.cl_tsr.class ).value.data ) );
        _argId++;
        return this;
    }

    /**
     *
     * @param value An int value which ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory patter).
     */
    public KernelCaller pass( int value ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_int, Pointer.to( new int[]{ value } ) );
        _argId++;
        return this;
    }

    /**
     *
     * @param value A float value which ought to be passed to the kernel.
     * @return This very KernelCaller instance (factory patter).
     */
    public KernelCaller pass( float value ) {
        clSetKernelArg( _kernel, _argId, Sizeof.cl_float, Pointer.to( new float[]{ value } ) );
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

    public void call( long[] globalWorkSizes, long[] localWorkSizes )
    {
        cl_event[] events = _getWaitList( _inputs.toArray( new Tsr[ 0 ] ) );
        if ( events.length > 0 ) {
            clWaitForEvents( events.length, events );
            _releaseEvents( _inputs.toArray( new Tsr[ 0 ] ) );
        }
        if ( localWorkSizes != null ) assert globalWorkSizes.length == localWorkSizes.length;
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

    @Contract( pure = true )
    private void _releaseEvents( @NotNull Tsr<Number>[] tensors ) {
        for ( Tsr<Number> t : tensors ) {
            if ( t.find( OpenCLDevice.cl_tsr.class ).value.event != null ) {
                clReleaseEvent(t.find( OpenCLDevice.cl_tsr.class ).value.event);
                t.find( OpenCLDevice.cl_tsr.class ).value.event = null;
            }
        }
    }

    @Contract( pure = true )
    private cl_event[] _getWaitList( @NotNull Tsr<Number>[] tensors ) {
        List<cl_event> list = new ArrayList<>();
        for ( Tsr<Number> t : tensors ) {
            cl_event event = t.find( OpenCLDevice.cl_tsr.class ).value.event;
            if ( event != null && !list.contains(event) ) {
                list.add( event );
            }
        }
        return list.toArray( new cl_event[ 0 ] );
    }

}
