package neureka.acceleration.opencl;

import static org.jocl.CL.*;

import java.lang.ref.Cleaner;
import java.lang.ref.ReferenceQueue;
import java.nio.*;
import java.util.*;

import neureka.Tsr;
import neureka.acceleration.AbstractDevice;
import neureka.acceleration.Device;
import neureka.acceleration.opencl.utility.WeakTensorReference;
import neureka.calculus.Function;
import neureka.utility.DataHelper;
import org.jocl.*;


public class OpenCLDevice extends AbstractDevice
{
    class cl_data{
        public cl_mem data;
        public int uses = 0;
        public int size = 0;
    }

    class cl_tsr{
        public int fp = 1;
        public cl_mem config;
        public cl_data value = new cl_data();
    }
    private Map<Object, cl_tsr> _mapping = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());

    private cl_device_id _did;

    public cl_device_id CLDeviceID(){
        return _did;
    }

    /**
     * The OpenCLPlaform
     */
    private OpenCLPlatform _platform;

    /**
     * The OpenCL command queue
     */
    private cl_command_queue _queue;

    private static Cleaner CLEANER = Cleaner.create();
    private ReferenceQueue _reference_queue;
    /**==============================================================================================================**/

    public OpenCLDevice(OpenCLPlatform platform, cl_device_id did)
    {
        _did = did;
        _platform = platform;
        //System.out.println(this.name());
        //System.out.println(this.type());
        // Create a command-queue for the selected device
        _queue = clCreateCommandQueue(platform.getContext(), did, 0, null);
        _reference_queue = new ReferenceQueue();

    }
    //---

    @Override
    public synchronized Collection<Tsr> tensors(){
        Collection<Object> collection = _mapping.keySet();
        Collection<Tsr> extracted = new ArrayList<>();
        collection.forEach((o)->{
            WeakTensorReference r = (WeakTensorReference) o;
            Tsr t = (Tsr)r.get();
            if(t!=null){
                extracted.add(t);
            }
        });
        return extracted;
    }

    @Override
    public void dispose() {
        _mapping.forEach((t, clt)->get((Tsr)((WeakTensorReference)t).get()));
        clFinish(_queue);
    }

    @Override
    public Device get(Tsr tensor)
    {
        double[] value = value64Of(tensor);
        rmv(tensor);
        if(tensor.has(Tsr.class)){
            Tsr gradient = (Tsr)tensor.find(Tsr.class);
            this.get(gradient);
        }
        tensor.setValue(value);
        return this;
    }

    @Override
    public Device add(Tsr tensor) {
        _add(tensor, 1, null);
        return this;
    }

    @Override
    public Device add(Tsr tensor, Tsr parent) {
        if(!parent.isOutsourced()){
            throw new IllegalStateException(
                    "[OpenClDevice][add(Tsr tensor, Tsr parent)]: Data parent is not outsourced!"
            );
        }
        _add(tensor, 1, _mapping.get(parent));
        return this;
    }

    private Device _add(Tsr tensor, int fp, cl_tsr parent)
    {
        cl_tsr newClt = new cl_tsr();
        //VALUE TRANSFER:
        if(parent==null){
            _store(tensor, newClt, fp, false);
            if(tensor.rqsGradient()&&tensor.has(Tsr.class)){
                //_store(tensor, newClt, fp, true);
                this.add(((Tsr)tensor.find(Tsr.class)));
            }
        } else {//tensor is a subset tensor of parent:
            newClt.fp = parent.fp;
            newClt.value = parent.value;
            newClt.value.uses ++;
        }
        //CONFIG TRANSFER: <[ shape | translation | idxmap | idx | scale ]>
        int rank = tensor.shape().length;
        int[] config = new int[rank*5];//3
        int[] shape = tensor.shape();
        //---
        for(int i=0; i<rank; i++) config[i] = shape[i];// -=> SHAPE
        //---
        int[] translation = tensor.translation();
        for(int i=rank; i<rank*2; i++) config[i] = translation[i-rank];// -=> TRANSLATION
        //---
        int[] idxmap = tensor.idxmap();
        for(int i=rank*2; i<rank*3; i++) config[i] = idxmap[i-rank*2];// -=> IDXMAP (translates scalar to dimension index)
        //---
        int[] idx = (int[])tensor.find(int[].class); //Look for additional configurations!
        if(idx!=null){
            for(int i=rank*3; i<rank*4; i++){// -=> IDX Basline (sliced tensors have this property)
                config[i] = idx[i-rank*3];
            }
        }
        //---
        // -=> IDX Baseline Scale (sliced tensors indexes might be scaled)
        for(int i=rank*4; i<rank*5; i++)config[i] = ((idx!=null)&&idx.length>tensor.rank())?idx[i-rank*3]:1;
        //---

        //SHAPE/TRANSLATION/IDXMAP/SCALEMAP TRANSFER:
        newClt.config = clCreateBuffer(
                _platform.getContext(),
                CL_MEM_READ_WRITE,
                config.length * Sizeof.cl_int,
                null, null
        );
        clEnqueueWriteBuffer(
                _queue,
                newClt.config,
                true, 0,
                config.length * Sizeof.cl_int,
                Pointer.to(config),
                0, null, null
        );
        cl_mem[] memos;
        if(tensor.rqsGradient()){
            memos = new cl_mem[]{newClt.value.data, newClt.config};//newClt.grad.data,
        } else {
            memos = new cl_mem[]{newClt.value.data, newClt.config};
        }
        int err = clEnqueueMigrateMemObjects(
                _queue,
                memos.length,
                memos,
                CL_MIGRATE_MEM_OBJECT_HOST,
                0,
                null,
                null
        );
        WeakTensorReference r = new WeakTensorReference(tensor, _reference_queue);
        _mapping.put(r, newClt);
        CLEANER.register(tensor, ()->{
            _rmv(r);
            //System.out.println("Garbage has been removed from GPU!!!");
        });
        tensor.add(this);
        if(tensor.isVirtual()){
            _execute_tensor_scalar(tensor, tensor.value64(0), Function.TYPES.LOOKUP.get("<"), -1);
        }
        tensor.setIsOutsourced(true);
        tensor.setIsVirtual(false);
        return this;
    }

    @Override
    public boolean has(Tsr tensor) {
        return _mapping.containsKey(tensor);
    }

    private void _store(Tsr tensor, cl_tsr newClTsr, int fp, boolean grd){
        Pointer p = null;
        int size = tensor.size();
        if(!tensor.isVirtual()){
            if(fp==1){
                float[] data = tensor.value32();
                data = (data==null)?new float[tensor.size()]:data;
                p = Pointer.to(data);
                size = data.length;
            } else {
                double[] data = tensor.value64();
                data = (data==null)?new double[tensor.size()]:data;
                p = Pointer.to(data);
                size = data.length;
            }
        }
        newClTsr.value.size = size;
        newClTsr.value.uses = 1;
        //VALUE TRANSFER:
        cl_mem mem = clCreateBuffer(
                _platform.getContext(),
                CL_MEM_READ_WRITE,
                size * Sizeof.cl_float*fp,
                null,
                null
        );
        newClTsr.value.data = mem;
        if(!tensor.isVirtual()){
            clEnqueueWriteBuffer(
                    _queue,
                    mem,
                    true, 0,
                    size * Sizeof.cl_float*fp,
                    p,
                    0, null, null
            );
        }
    }


    @Override
    public Device rmv(Tsr tensor) {
        cl_tsr clt = _mapping.get(tensor);
        if(clt==null) return this; //THIS SHOULD NOT BE?
        clReleaseMemObject(clt.config);//remove translations/shapes from device!
        clReleaseMemObject(clt.value.data);
        _mapping.remove(tensor);
        tensor.setIsOutsourced(false);
        return this;
    }
    private void _rmv(WeakTensorReference reference){
        cl_tsr clt = _mapping.get(reference);
        clReleaseMemObject(clt.config);//remove translations/shapes from device!
        if(clt.value.uses<=1){
            clReleaseMemObject(clt.value.data);
        } else {
            clt.value.uses --;
        }
        _mapping.remove(reference);
    }

    @Override
    public Device overwrite64(Tsr tensor, double[] value) {//TODO: Make value an object!
        cl_tsr clt = _mapping.get(tensor);
        if(clt.fp==1){
            overwrite32(tensor, DataHelper.doubleToFloat(value));
        } else {
            clEnqueueWriteBuffer(
                    _queue,
                    clt.value.data,
                    CL_TRUE,
                    0,
                    Sizeof.cl_double * value.length,
                    Pointer.to(value),
                    0,
                    null,
                    null
            );
        }
        return this;
    }

    @Override
    public Device overwrite32(Tsr tensor, float[] value) {//TODO: Make value an object!
        cl_tsr clt = _mapping.get(tensor);
        if(clt.fp==1){
            clEnqueueWriteBuffer(
                    _queue,
                    clt.value.data,
                    CL_TRUE,
                    0,
                    Sizeof.cl_float * value.length,
                    Pointer.to(value),
                    0,
                    null,
                    null
            );
        } else {
           overwrite64(tensor, DataHelper.floatToDouble(value));
        }
        return this;
    }

    @Override
    public Device swap(Tsr former, Tsr replacement) {
        cl_tsr clTsr = _mapping.get(former);
        _mapping.remove(former);
        WeakTensorReference r = new WeakTensorReference(replacement, _reference_queue);
        _mapping.put(r, clTsr);
        return this;
    }

    @Override
    public double[] value64Of(Tsr tensor) {
        cl_tsr clt = _mapping.get(tensor);
        if(clt.fp==1){
            return DataHelper.floatToDouble(value32Of(tensor));
        } else {
            double[] data = new double[clt.value.size];//tensor.size()];
            clEnqueueReadBuffer(
                    _queue,
                    clt.value.data,//(grd)?clt.grad.data:clt.value.data,
                    CL_TRUE,
                    0,
                    Sizeof.cl_double * data.length,
                    Pointer.to(data),
                    0,
                    null,
                    null
            );
            return data;
        }
    }

    @Override
    public float[] value32Of(Tsr tensor){
        cl_tsr clt = _mapping.get(tensor);
        if(clt.fp==1){
            float[] data = new float[clt.value.size];//tensor.size()];
            clEnqueueReadBuffer(
                    _queue,
                    clt.value.data,//(grd)?clt.grad.data:clt.value.data,
                    CL_TRUE,
                    0,
                    Sizeof.cl_float * data.length,
                    Pointer.to(data),
                    0,
                    null,
                    null
            );
            return data;
        } else {
            return DataHelper.doubleToFloat(value64Of(tensor));
        }
    }

    @Override
    protected void _enqueue(Tsr[] tsrs, int d, int f_id){

        switch (Function.TYPES.REGISTER[f_id]) {
            case "x":
                if (d >= 0) {
                    if (d == 0) tsrs[0] = tsrs[2]; else tsrs[0] = tsrs[1];
                    return;
                } else {
                    tsrs = new Tsr[]{tsrs[0], tsrs[1], tsrs[2]};
                }
                break;
            case ("x" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};break;
            case ("a" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};break;
            case ("s" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};break;
            case ("d" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};break;
            case ("p" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};break;
            case ("m" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};break;
            case ">": tsrs = new Tsr[]{tsrs[1], tsrs[0]};break;
        }





        int gwz = (tsrs[0]!=null)?tsrs[0].size():tsrs[1].size();
        int offset = (tsrs[0]!=null)?0:1;
        String chosen = _platform.kernelNameOf(f_id);
        //System.out.println("chosen enq: "+chosen);
        cl_kernel kernel = _platform.getKernels().get(chosen);

        cl_mem drn = _mapping.get(tsrs[offset]).value.data;
        cl_mem src1 = _mapping.get(tsrs[offset+1]).value.data;
        if(Function.TYPES.isFunction(f_id)){
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(drn));//=> drain
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(_mapping.get(tsrs[offset]).config));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(src1));//=>src1
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(_mapping.get(tsrs[offset+1]).config));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{ tsrs[0].rank() }));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{ d }));
        } else {
            //TODO: Function.TYPES.rqsAdditionalArgument(f_id)
            cl_mem src2 = _mapping.get(tsrs[offset+2]).value.data;
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(drn));//=> drain
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(_mapping.get(tsrs[offset]).config));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(src1));//=>src1
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(_mapping.get(tsrs[offset+1]).config));
            clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(src2));//=>src2
            clSetKernelArg(kernel, 5, Sizeof.cl_mem, Pointer.to(_mapping.get(tsrs[offset+2]).config));
            clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{ tsrs[0].rank() }));
            clSetKernelArg(kernel, 7, Sizeof.cl_int, Pointer.to(new int[]{ d }));
        }
        clEnqueueNDRangeKernel(
                _queue, kernel,
                1,
                null,
                new long[]{gwz},
                null,
                0,
                null,
                null
        );
    }

    @Override
    protected void _enqueue(Tsr t, double value, int d, int f_id){
        int gwz = t.size();
        String chosen =  "scalar_"+_platform.kernelNameOf(f_id).replace("operate_", "");
        cl_kernel kernel = _platform.getKernels().get(chosen);
        cl_mem drn = _mapping.get(t).value.data;
        cl_mem src1 = _mapping.get(t).value.data;
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(drn));//=> drain
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(_mapping.get(t).config));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(src1));//=>src1
        clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(_mapping.get(t).config));
        clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[]{(float)value}));
        clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{ t.rank() }));
        clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{ d }));
        clEnqueueNDRangeKernel(
                _queue, kernel,
                1,
                null,
                new long[]{gwz},
                null,
                0,
                null,
                null
        );
    }


    public String name(){
        return DeviceQuery.getString(_did, CL_DEVICE_NAME);
    }

    public String vendor(){
        return DeviceQuery.getString(_did, CL_DEVICE_VENDOR);
    }

    public String version(){
        return DeviceQuery.getString(_did, CL_DRIVER_VERSION);
    }

    public String type(){
        long deviceType = DeviceQuery.getLong(_did, CL_DEVICE_TYPE);
        if( (deviceType & CL_DEVICE_TYPE_CPU) != 0)
            return "CPU";
        if( (deviceType & CL_DEVICE_TYPE_GPU) != 0)
            return "GPU";
        if( (deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0)
            return "ACCELERATOR";
        if( (deviceType & CL_DEVICE_TYPE_DEFAULT) != 0)
            return "DEFAULT";
        return "UNKNOWN";
    }

    public int maxComputeUnits(){
        return DeviceQuery.getInt(_did, CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    public long maxWorkItemSimensions(){
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    }

    public long[] maxWorkItemSizes(){
        return DeviceQuery.getSizes(_did, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
    }

    public long maxWorkGroupSize(){
        return DeviceQuery.getSize(_did, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    public long maxClockFrequenzy(){
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    public int maxAddressBits(){
        return DeviceQuery.getInt(_did, CL_DEVICE_ADDRESS_BITS);
    }

    public long maxMemAllocSize(){
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    }

    public long globalMemSize(){
        return DeviceQuery.getLong(_did, CL_DEVICE_GLOBAL_MEM_SIZE);
    }

    public int errorCorrectionSupport(){
        return DeviceQuery.getInt(_did, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
    }

    public int localMemType(){
        return DeviceQuery.getInt(_did, CL_DEVICE_LOCAL_MEM_TYPE);
    }

    public long localMemSize(){
        return DeviceQuery.getLong(_did, CL_DEVICE_LOCAL_MEM_SIZE);
    }

    public long maxConstantBufferSize(){
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    }

    public long maxConstantBufferSizeKB(){
        return (int)(DeviceQuery.getLong(_did, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)/1024);
    }

    public boolean queueExecIsOrdered(){
        long queueProperties = DeviceQuery.getLong(_did, CL_DEVICE_QUEUE_PROPERTIES);
        return (( queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ) != 0);
    }

    public boolean queueProfilingIsEnabled(){
        long queueProperties = DeviceQuery.getLong(_did, CL_DEVICE_QUEUE_PROPERTIES);
        return (( queueProperties & CL_QUEUE_PROFILING_ENABLE ) != 0);
    }

    public int imageSupport(){
        return DeviceQuery.getInt(_did, CL_DEVICE_IMAGE_SUPPORT);
    }

    public int maxReadImageArgs(){
        return DeviceQuery.getInt(_did, CL_DEVICE_MAX_READ_IMAGE_ARGS);
    }

    public int maxWriteImageArgs(){
       return DeviceQuery.getInt(_did, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
    }

    public long singleFPConfig(){
        return DeviceQuery.getLong(_did, CL_DEVICE_SINGLE_FP_CONFIG);
    }

    public long image2DMaxWidth(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE2D_MAX_WIDTH);
    }

    public long image2DMaxHeight(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
    }

    public long image3DMaxWidth(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_WIDTH);
    }

    public long image3DMaxHeight(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
    }

    public long image3DMaxDepth(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_DEPTH);
    }

    public int prefVecWidthChar(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
    }

    public int prefVecWidthShort(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
    }

    public int prefVecWidthInt(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
    }

    public int prefVecWidthLong(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
    }

    public int prefVecWidthFloat(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
    }

    public int prefVecWidthDouble(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
    }

    public static class DeviceQuery
    {
        /**
         * Returns the value64 of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static int getInt(cl_device_id device, int paramName)
        {
            return getInts(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value64
         */
        public static int[] getInts(cl_device_id device, int paramName, int numValues)
        {
            int values[] = new int[numValues];
            clGetDeviceInfo(device, paramName, Sizeof.cl_int * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         * Returns the value64 of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static long getLong(cl_device_id device, int paramName)
        {
            return getLongs(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value64
         */
        public static long[] getLongs(cl_device_id device, int paramName, int numValues)
        {
            long values[] = new long[numValues];
            clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         * Returns the value64 of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static String getString(cl_device_id device, int paramName)
        {
            // Obtain the length of the string that will be queried
            long size[] = new long[1];
            clGetDeviceInfo(device, paramName, 0, null, size);

            // Create a buffer of the appropriate size and fill it with the info
            byte buffer[] = new byte[(int)size[0]];
            clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

            // Create a string from the buffer (excluding the trailing \0 byte)
            return new String(buffer, 0, buffer.length-1);
        }

        /**
         * Returns the value64 of the platform info parameter with the given name
         *
         * @param platform The platform
         * @param paramName The parameter name
         * @return The value64
         */
        public static String getString(cl_platform_id platform, int paramName)
        {
            // Obtain the length of the string that will be queried
            long size[] = new long[1];
            clGetPlatformInfo(platform, paramName, 0, null, size);

            // Create a buffer of the appropriate size and fill it with the info
            byte buffer[] = new byte[(int)size[0]];
            clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);

            // Create a string from the buffer (excluding the trailing \0 byte)
            return new String(buffer, 0, buffer.length-1);
        }

        /**
         * Returns the value64 of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static long getSize(cl_device_id device, int paramName)
        {
            return getSizes(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value64
         */
        public static long[] getSizes(cl_device_id device, int paramName, int numValues)
        {
            // The size of the returned data has to depend on
            // the size of a size_t, which is handled here
            ByteBuffer buffer = ByteBuffer.allocate(
                    numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
            clGetDeviceInfo(device, paramName, Sizeof.size_t * numValues,
                    Pointer.to(buffer), null);
            long values[] = new long[numValues];
            if (Sizeof.size_t == 4)
            {
                for (int i=0; i<numValues; i++)
                {
                    values[i] = buffer.getInt(i * Sizeof.size_t);
                }
            }
            else
            {
                for (int i=0; i<numValues; i++)
                {
                    values[i] = buffer.getLong(i * Sizeof.size_t);
                }
            }
            return values;
        }

    }


}
