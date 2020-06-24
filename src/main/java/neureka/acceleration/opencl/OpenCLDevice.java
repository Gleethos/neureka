package neureka.acceleration.opencl;

import static org.jocl.CL.*;

import java.nio.*;
import java.util.*;

import neureka.Component;
import neureka.Tsr;
import neureka.acceleration.AbstractDevice;
import neureka.acceleration.Device;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;
import neureka.calculus.environment.executors.*;
import neureka.utility.DataHelper;
import org.jocl.*;

public class OpenCLDevice extends AbstractDevice
{
    static class cl_value {
        public cl_mem data;
        public int size = 0;
        public cl_event event;
    }

    static class cl_config {
        public cl_mem data;
    }

    static class cl_tsr implements Component<Tsr> {
        public int fp = 1;
        public cl_config config = new cl_config();// Tensor configurations are always unique!
        public cl_value value;

        @Override
        public void update(Tsr oldOwner, Tsr newOwner) {
            // Update not needed....
        }
    }

    private final Set<Tsr> _tensors = Collections.newSetFromMap(new WeakHashMap<Tsr, Boolean>());

    private final cl_device_id _did;

    public cl_device_id CLDeviceID() {
        return _did;
    }

    /**
     * The OpenCLPlaform
     */
    private final OpenCLPlatform _platform;

    /**
     * The OpenCL command queue
     */
    private final cl_command_queue _queue;

    //==================================================================================================================

    /**
     * @param platform
     * @param did
     */
    private OpenCLDevice(OpenCLPlatform platform, cl_device_id did)
    {
        _did = did;
        _platform = platform;
        _queue = clCreateCommandQueueWithProperties(// Create a command-queue for the selected device
                platform.getContext(), did,
                null,
                null
        );
        //Runtime.getRuntime().addShutdownHook(new Thread(()->{
        //    _mapping.forEach((k, v)->{
        //        if(v.value.event!=null) clWaitForEvents(1, new cl_event[]{v.value.event});
        //        clReleaseMemObject(v.config);
        //        clReleaseMemObject(v.value.data);
        //    });
        //    clReleaseCommandQueue(_queue);
        //    clReleaseContext(_context);
        //}));
    }

    public static OpenCLDevice instance(OpenCLPlatform platform, cl_device_id did){
        if(!platform.has(did)) platform.put(did,  new OpenCLDevice(platform, did));
        return platform.get(did);
    }

    /**
     * @return A collection of all tensors currently stored on the device.
     */
    @Override
    public synchronized Collection<Tsr> tensors() {
        Collection<Collection<Tsr>> collection = Collections.singleton(_tensors);
        Collection<Tsr> extracted = new ArrayList<>();
        collection.forEach(c -> c.forEach(t->{ if (t != null) extracted.add(t); }));
        return extracted;
    }

    @Override
    public void dispose() {
        _tensors.forEach(this::get);
        clFinish(_queue);
    }

    @Override
    public Device get(Tsr tensor) {
        double[] value = value64Of(tensor);
        rmv(tensor);
        tensor.forComponent(Tsr.class, this::get);
        tensor.setValue(value);
        return this;
    }

    @Override
    public Device add(Tsr tensor) {
        _add(tensor, null);
        return this;
    }

    @Override
    public Device add(Tsr tensor, Tsr parent) {
        if (!parent.isOutsourced()) throw new IllegalStateException("Data parent is not outsourced!");
        _add(tensor, parent.find(cl_tsr.class));
        _tensors.add(tensor);
        tensor.add(this);
        return this;
    }

    private void _add(Tsr tensor, cl_tsr parent) {
        cl_tsr newClt = new cl_tsr();
        {
            final cl_mem clConfMem = newClt.config.data;
            _cleaning(newClt.config, () ->clReleaseMemObject(clConfMem));
        }
        //VALUE TRANSFER:
        if (parent == null) {
            newClt.value = new cl_value();
            _store(tensor, newClt, 1);
            if (tensor.rqsGradient() && tensor.has(Tsr.class)) {
                this.add(tensor.find(Tsr.class));
            }
            {
                final cl_mem clValMem = newClt.value.data;
                cl_event clValEvent = newClt.value.event;
                _cleaning(newClt.value, () -> {
                    if(clValEvent!=null) clWaitForEvents(1, new cl_event[]{clValEvent});
                    clReleaseMemObject(clValMem);//Removing value.. from device!
                });
            }
        } else {//tensor is a subset tensor of parent:
            newClt.fp = parent.fp;
            newClt.value = parent.value;
        }
        //CONFIG TRANSFER: <[ shape | translation | idxmap | idx | scale ]>
        int rank = tensor.rank();
        int[] config = new int[rank * 5];
        System.arraycopy(tensor.getNDConf().shape(), 0, config, 0, rank);// -=> SHAPE COPY
        System.arraycopy(tensor.getNDConf().translation(), 0, config, rank * 1, rank);// -=> TRANSLATION COPY
        System.arraycopy(tensor.getNDConf().idxmap(), 0, config, rank * 2, rank);// -=> IDXMAP COPY (translates scalar to dimension index)
        System.arraycopy(tensor.getNDConf().offset(), 0, config, rank * 3, rank);// -=> SPREAD
        System.arraycopy(tensor.getNDConf().spread(), 0, config, rank * 4, rank);

        //SHAPE/TRANSLATION/IDXMAP/OFFSET/SPREAD TRANSFER:
        newClt.config.data = clCreateBuffer(
                _platform.getContext(),
                CL_MEM_READ_WRITE,
                config.length * Sizeof.cl_int,
                null, null
        );
        clEnqueueWriteBuffer(
                _queue,
                newClt.config.data,
                CL_TRUE,
                0,
                config.length * Sizeof.cl_int,
                Pointer.to(config),
                0,
                null,
                null
        );
        cl_mem[] memos;
        memos = new cl_mem[]{newClt.value.data, newClt.config.data};

        clEnqueueMigrateMemObjects(
                _queue,
                memos.length,
                memos,
                CL_MIGRATE_MEM_OBJECT_HOST,
                0,
                null,
                null
        );

        _tensors.add(tensor);

        tensor.add(newClt);
        tensor.add(this);
        if (tensor.isVirtual()) _enqueue(tensor, tensor.value64(0), -1, OperationType.instance("<"));
        tensor.setIsOutsourced(true);
        tensor.setIsVirtual(false);
    }

    @Override
    public boolean has(Tsr tensor) {
        return _tensors.contains(tensor);
    }

    private void _store(Tsr tensor, cl_tsr newClTsr, int fp) {
        Pointer p = null;
        int size = tensor.size();
        if (!tensor.isVirtual()) {
            if (fp == 1) {
                float[] data = tensor.value32();
                data = (data == null) ? new float[tensor.size()] : data;
                p = Pointer.to(data);
                size = data.length;
            } else {
                double[] data = tensor.value64();
                data = (data == null) ? new double[tensor.size()] : data;
                p = Pointer.to(data);
                size = data.length;
            }
        }
        newClTsr.value.size = size;
        //VALUE TRANSFER:
        cl_mem mem = clCreateBuffer(
                _platform.getContext(),
                CL_MEM_READ_WRITE,
                size * (long)Sizeof.cl_float * fp,
                null,
                null
        );
        newClTsr.value.data = mem;
        if (!tensor.isVirtual()) {
            clEnqueueWriteBuffer(
                    _queue,
                    mem,
                    CL_TRUE,
                    0,
                    size * (long)Sizeof.cl_float * fp,
                    p,
                    0,
                    null,
                    null
            );
        }
    }


    @Override
    public Device rmv(Tsr tensor) {
        cl_tsr clt = tensor.find(cl_tsr.class);
        if (clt == null) return this;
        _tensors.remove(tensor);
        tensor.setIsOutsourced(false);
        tensor.remove(cl_tsr.class);
        return this;
    }

    //private void _rmv(WeakReference<Tsr> reference) {
    //    cl_tsr clt = _mapping.get(reference);
    //    clReleaseMemObject(clt.config);//remove translations/shapes/spread/offset... from device!
    //    //if (clt.value.uses <= 1){
    //    //    clReleaseMemObject(clt.value.data);
    //    //} else clt.value.uses--;
    //    _mapping.remove(reference);
    //}

    @Override
    public Device overwrite64(Tsr tensor, double[] value) {
        cl_tsr clt = tensor.find(cl_tsr.class);
        if (clt.fp == 1) {
            overwrite32(tensor, DataHelper.doubleToFloat(value));
        } else {
            if(clt.value.event!=null) clWaitForEvents(1, new cl_event[]{clt.value.event});
            clt.value.event = new cl_event();
            clEnqueueWriteBuffer(
                    _queue,
                    clt.value.data,
                    CL_FALSE,
                    0,
                    Sizeof.cl_double * value.length,
                    Pointer.to(value),
                    0,
                    null,
                    clt.value.event
            );
        }
        return this;
    }

    private void _releaseEvents(Tsr[] tsrs){
        for(Tsr t : tsrs){
            if( t.find(cl_tsr.class).value.event != null ){
                clReleaseEvent(t.find(cl_tsr.class).value.event);
                t.find(cl_tsr.class).value.event = null;
            }
        }
    }

    private cl_event[] _getWaitList(Tsr[] tsrs){
        List<cl_event> list = new ArrayList<>();
        for (Tsr t : tsrs) {
            cl_event event = t.find(cl_tsr.class).value.event;
            if (event != null && !list.contains(event)) {
                list.add(event);
            }
        }
        return list.toArray(new cl_event[0]);
    }

    @Override
    public Device overwrite32(Tsr tensor, float[] value) {
        cl_tsr clt = tensor.find(cl_tsr.class);
        if (clt.fp == 1) {
            if(clt.value.event!=null){
                clWaitForEvents(1, new cl_event[]{clt.value.event});
            }
            clt.value.event = new cl_event();
            clEnqueueWriteBuffer(
                    _queue,
                    clt.value.data,
                    CL_TRUE,
                    0,
                    Sizeof.cl_float * value.length,
                    Pointer.to(value),
                    0,
                    null,
                    clt.value.event
            );
        } else {
            overwrite64(tensor, DataHelper.floatToDouble(value));
        }
        return this;
    }

    @Override
    public Device swap(Tsr former, Tsr replacement) {
        cl_tsr clTsr = former.find(cl_tsr.class);
        former.remove(cl_tsr.class);
        replacement.add(clTsr);
        _tensors.remove(former);
        _tensors.add(replacement);
        return this;
    }

    @Override
    public double[] value64Of(Tsr tensor) {
        cl_tsr clt = tensor.find(cl_tsr.class);
        if (clt.fp == 1) {
            return DataHelper.floatToDouble(value32Of(tensor));
        } else {
            double[] data = new double[clt.value.size];
            clEnqueueReadBuffer(
                    _queue,
                    clt.value.data,
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
    public float[] value32Of(Tsr tensor) {
        cl_tsr clt = tensor.find(cl_tsr.class);
        if (clt.fp == 1) {
            float[] data = new float[clt.value.size];
            clEnqueueReadBuffer(
                    _queue,
                    clt.value.data,
                    CL_TRUE,
                    0,
                    (long)Sizeof.cl_float * data.length,
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

    public KernelBuilder getKernel(OperationTypeImplementation.ExecutionCall call){
        String chosen = _platform.kernelNameOf(call.getType());
        cl_kernel kernel = _platform.getKernels().get(chosen);
        return new KernelBuilder(kernel, _queue);
    }

    @Override
    protected void _enqueue(Tsr[] tsrs, int d, OperationType type)
    {
        int offset = (tsrs[0] != null) ? 0 : 1;
        int gwz = (tsrs[0] != null) ? tsrs[0].size() : tsrs[1].size();
        String chosen = _platform.kernelNameOf(type);
        cl_kernel kernel = _platform.getKernels().get(chosen);

        if (type.supportsImplementation(Activation.class) && !type.isIndexer()) {
            new KernelBuilder(kernel, _queue)
                    .pass(tsrs[offset])
                    .pass(tsrs[offset + 1])
                    .pass(tsrs[0].rank())
                    .pass(d)
                    .call(gwz);
        } else {
            new KernelBuilder(kernel, _queue)
                    .pass(tsrs[offset])
                    .pass(tsrs[offset + 1])
                    .pass(tsrs[offset + 2])
                    .pass(tsrs[0].rank())
                    .pass(d)
                    .call(gwz);
        }
    }


    @Override
    protected void _enqueue(Tsr t, double value, int d, OperationType type) {
        int gwz = t.size();
        String chosen = "scalar_" + _platform.kernelNameOf(type).replace("operate_", "");
        cl_kernel kernel = _platform.getKernels().get(chosen);
        new KernelBuilder(kernel, _queue)
                .pass(t)
                .pass(t)
                .pass((float)value)
                .pass(t.rank())
                .pass(d)
                .call(gwz);
    }


    public String name() {
        return DeviceQuery.getString(_did, CL_DEVICE_NAME);
    }

    public String vendor() {
        return DeviceQuery.getString(_did, CL_DEVICE_VENDOR);
    }

    public String version() {
        return DeviceQuery.getString(_did, CL_DRIVER_VERSION);
    }

    public String type() {
        long deviceType = DeviceQuery.getLong(_did, CL_DEVICE_TYPE);
        if ((deviceType & CL_DEVICE_TYPE_CPU) != 0)
            return "CPU";
        if ((deviceType & CL_DEVICE_TYPE_GPU) != 0)
            return "GPU";
        if ((deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0)
            return "ACCELERATOR";
        if ((deviceType & CL_DEVICE_TYPE_DEFAULT) != 0)
            return "DEFAULT";
        return "UNKNOWN";
    }

    public int maxComputeUnits() {
        return DeviceQuery.getInt(_did, CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    public long maxWorkItemSimensions() {
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    }

    public long[] maxWorkItemSizes() {
        return DeviceQuery.getSizes(_did, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
    }

    public long maxWorkGroupSize() {
        return DeviceQuery.getSize(_did, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    public long maxClockFrequenzy() {
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    public int maxAddressBits() {
        return DeviceQuery.getInt(_did, CL_DEVICE_ADDRESS_BITS);
    }

    public long maxMemAllocSize() {
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    }

    public long globalMemSize() {
        return DeviceQuery.getLong(_did, CL_DEVICE_GLOBAL_MEM_SIZE);
    }

    public int errorCorrectionSupport() {
        return DeviceQuery.getInt(_did, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
    }

    public int localMemType() {
        return DeviceQuery.getInt(_did, CL_DEVICE_LOCAL_MEM_TYPE);
    }

    public long localMemSize() {
        return DeviceQuery.getLong(_did, CL_DEVICE_LOCAL_MEM_SIZE);
    }

    public long maxConstantBufferSize() {
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    }

    public long maxConstantBufferSizeKB() {
        return (int) (DeviceQuery.getLong(_did, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) / 1024);
    }
    /*
    public boolean queueExecIsOrdered() {
        long queueProperties = DeviceQuery.getLong(_did, CL_DEVICE_QUEUE_PROPERTIES);//Deprecation!
        return ((queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0);
    }

    public boolean queueProfilingIsEnabled() {
        long queueProperties = DeviceQuery.getLong(_did, CL_DEVICE_QUEUE_PROPERTIES);
        return ((queueProperties & CL_QUEUE_PROFILING_ENABLE) != 0);
    }
    */
    public int imageSupport() {
        return DeviceQuery.getInt(_did, CL_DEVICE_IMAGE_SUPPORT);
    }

    public int maxReadImageArgs() {
        return DeviceQuery.getInt(_did, CL_DEVICE_MAX_READ_IMAGE_ARGS);
    }

    public int maxWriteImageArgs() {
        return DeviceQuery.getInt(_did, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
    }

    public long singleFPConfig() {
        return DeviceQuery.getLong(_did, CL_DEVICE_SINGLE_FP_CONFIG);
    }

    public long image2DMaxWidth() {
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE2D_MAX_WIDTH);
    }

    public long image2DMaxHeight() {
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
    }

    public long image3DMaxWidth() {
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_WIDTH);
    }

    public long image3DMaxHeight() {
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
    }

    public long image3DMaxDepth() {
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_DEPTH);
    }

    public int prefVecWidthChar() {
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
    }

    public int prefVecWidthShort() {
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
    }

    public int prefVecWidthInt() {
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
    }

    public int prefVecWidthLong() {
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
    }

    public int prefVecWidthFloat() {
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
    }

    public int prefVecWidthDouble() {
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
    }

    public static class DeviceQuery {
        /**
         * Returns the value64 of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static int getInt(cl_device_id device, int paramName) {
            return getInts(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value64
         */
        public static int[] getInts(cl_device_id device, int paramName, int numValues) {
            int[] values = new int[numValues];
            clGetDeviceInfo(device, paramName, (long)Sizeof.cl_int * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         * Returns the value64 of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static long getLong(cl_device_id device, int paramName) {
            return getLongs(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value64
         */
        public static long[] getLongs(cl_device_id device, int paramName, int numValues) {
            long[] values = new long[numValues];
            clGetDeviceInfo(device, paramName, (long)Sizeof.cl_long * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         * Returns the value64 of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static String getString(cl_device_id device, int paramName) {
            // Obtain the length of the string that will be queried
            long[] size = new long[1];
            clGetDeviceInfo(device, paramName, 0, null, size);

            // Create a buffer of the appropriate size and fill it with the info
            byte[] buffer = new byte[(int) size[0]];
            clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

            // Create a string from the buffer (excluding the trailing \0 byte)
            return new String(buffer, 0, buffer.length - 1);
        }

        /**
         * Returns the value64 of the platform info parameter with the given name
         *
         * @param platform  The platform
         * @param paramName The parameter name
         * @return The value64
         */
        public static String getString(cl_platform_id platform, int paramName) {
            // Obtain the length of the string that will be queried
            long[] size = new long[1];
            clGetPlatformInfo(platform, paramName, 0, null, size);

            // Create a buffer of the appropriate size and fill it with the info
            byte[] buffer = new byte[(int) size[0]];
            clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);

            // Create a string from the buffer (excluding the trailing \0 byte)
            return new String(buffer, 0, buffer.length - 1);
        }

        /**
         * Returns the value64 of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static long getSize(cl_device_id device, int paramName) {
            return getSizes(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value64
         */
        public static long[] getSizes(cl_device_id device, int paramName, int numValues) {
            // The size of the returned data has to depend on
            // the size of a size_t, which is handled here
            ByteBuffer buffer = ByteBuffer.allocate(
                    numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
            clGetDeviceInfo(device, paramName, (long)Sizeof.size_t * numValues,
                    Pointer.to(buffer), null);
            long[] values = new long[numValues];
            if (Sizeof.size_t == 4) {
                for (int i = 0; i < numValues; i++) {
                    values[i] = buffer.getInt(i * Sizeof.size_t);
                }
            } else {
                for (int i = 0; i < numValues; i++) {
                    values[i] = buffer.getLong(i * Sizeof.size_t);
                }
            }
            return values;
        }

    }


}
