package neureka.devices.opencl;

import neureka.Tsr;
import org.jocl.*;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

public class KernelBuilder
{
    private final cl_command_queue _queue;
    private final cl_kernel _kernel;
    private int _argId;
    private final List<Tsr> _inputs;

    public KernelBuilder(cl_kernel kernel, cl_command_queue queue){
        _queue = queue;
        _kernel = kernel;
        _inputs = new ArrayList<>();
        _argId = 0;
    }

    /**
     * This method passes 2 arguments to the kernel.
     * One for the data of the tensor and one for the configuration data!
     * @param t
     * @return
     */
    public KernelBuilder pass(Tsr<Number> t){
        _inputs.add(t);
        clSetKernelArg(_kernel, _argId, Sizeof.cl_mem, Pointer.to(t.find(OpenCLDevice.cl_tsr.class).value.data));
        _argId++;
        clSetKernelArg(_kernel, _argId, Sizeof.cl_mem, Pointer.to(t.find(OpenCLDevice.cl_tsr.class).config.data));
        _argId++;
        return this;
    }

    public KernelBuilder pass(int i) {
        clSetKernelArg(_kernel, _argId, Sizeof.cl_int, Pointer.to(new int[]{i}));
        _argId++;
        return this;
    }

    public KernelBuilder pass(float f){
        clSetKernelArg(_kernel, _argId, Sizeof.cl_float, Pointer.to(new float[]{f}));
        _argId++;
        return this;
    }

    public void call(int globalWorkSize)
    {
        cl_event[] events = _getWaitList(_inputs.toArray(new Tsr[ 0 ]));
        if(events.length>0){
            clWaitForEvents(events.length, events);
            _releaseEvents(_inputs.toArray(new Tsr[ 0 ]));
        }
        clEnqueueNDRangeKernel(
                _queue, _kernel,
                1,
                null,
                new long[]{globalWorkSize},
                null,
                0,
                null,
                null
        );
    }

    private void _releaseEvents(Tsr[] tsrs){
        for(Tsr<Number> t : tsrs){
            if( t.find(OpenCLDevice.cl_tsr.class).value.event != null ){
                clReleaseEvent(t.find(OpenCLDevice.cl_tsr.class).value.event);
                t.find(OpenCLDevice.cl_tsr.class).value.event = null;
            }
        }
    }

    private cl_event[] _getWaitList(Tsr[] tsrs){
        List<cl_event> list = new ArrayList<>();
        for (Tsr<Number> t : tsrs) {
            cl_event event = t.find(OpenCLDevice.cl_tsr.class).value.event;
            if (event != null && !list.contains(event)) {
                list.add(event);
            }
        }
        return list.toArray(new cl_event[ 0 ]);
    }

}
