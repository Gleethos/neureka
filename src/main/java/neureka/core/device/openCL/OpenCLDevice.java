package neureka.core.device.openCL;

import static org.jocl.CL.*;

import java.nio.*;
import java.util.*;

import neureka.core.Tsr;
import neureka.core.device.IDevice;
import neureka.core.function.IFunction;
import neureka.core.function.factory.Function;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.utility.DataHelper;
import org.jocl.*;


public class OpenCLDevice implements IDevice
{
    class cl_tsr{
        public int fp = 1;
        public cl_mem config;
        public cl_mem value;
        public cl_mem grad;
    }

    private Map<Tsr, cl_tsr> _mapping = new HashMap<>();

    private cl_device_id _did;

    /**
     * The OpenCLPlaform
     */
    private OpenCLPlatform _platform;

    /**
     * The OpenCL command queue
     */
    private cl_command_queue _queue;


    /**==============================================================================================================**/

    public OpenCLDevice(OpenCLPlatform platform, cl_device_id did)
    {
        _did = did;
        _platform = platform;
        System.out.println(this.name());
        System.out.println(this.type());
        // Create a command-queue for the selected device
        _queue = clCreateCommandQueue(platform.getContext(), did, 0, null);
    }

    //---

    @Override
    public void dispose() {
        _mapping.forEach((t, clt)->get(t));
        clFinish(_queue);
    }

    @Override
    public IDevice get(Tsr tensor) {
        tensor.setValue(value64Of(tensor, false));
        if(tensor.rqsGradient()){
            tensor.setValue(value64Of(tensor, true));
        }
        rmv(tensor);
        return this;
    }

    @Override
    public IDevice add(Tsr tensor) {
        add(tensor, 1);
        return this;
    }

    @Override
    public boolean has(Tsr tensor) {
        return _mapping.containsKey(tensor);
    }

    private void _store(Tsr tensor, cl_tsr newClTsr, int fp, boolean grd){
        //cl_tsr newClTsr = new cl_tsr();
        //tensor.setIsVirtual(false);//TODO: optimize: if virtual: copy !efficiently
        Pointer p;
        int size;
        if(fp==1){
            float[] data = (grd)?tensor.gradient32():tensor.value32();
            data = (data==null)?new float[tensor.size()]:data;
            p = Pointer.to(data);
            size = data.length;
        } else {
            double[] data = (grd)?tensor.gradient64():tensor.value64();
            data = (data==null)?new double[tensor.size()]:data;
            p = Pointer.to(data);
            size = data.length;
        }
        //VALUE TRANSFER:
        cl_mem mem = clCreateBuffer(
                _platform.getContext(),
                CL_MEM_READ_WRITE,
                size * Sizeof.cl_float*fp,
                null,
                null
        );
        if(grd){
            newClTsr.grad = mem;
        } else {
            newClTsr.value = mem;
        }
        clEnqueueWriteBuffer(
                _queue,
                mem,
                true, 0,
                size * Sizeof.cl_float*fp,
                p,
                0, null, null
        );
    }

    public IDevice add(Tsr tensor, int fp) {
        tensor.setIsVirtual(false);//TODO: optimize: if virtual: copy !efficiently
        cl_tsr newClt = new cl_tsr();
        //VALUE TRANSFER:
        _store(tensor, newClt, fp, false);
        if(tensor.rqsGradient()){
            _store(tensor, newClt, fp, true);
        }
        int rank = tensor.shape().length;
        int[] config = new int[rank*3];
        int[] shape = tensor.shape();
        for(int i=0; i<rank; i++){
            config[i] = shape[i];
        }
        int[] translation = tensor.translation();
        for(int i=rank; i<rank*2; i++){
            config[i] = translation[i-rank];
        }
        int[] idxmap = tensor.idxmap();
        for(int i=rank*2; i<rank*3; i++){
            config[i] = idxmap[i-rank*2];
        }
        //SHAPE/TRANSLATION/IDXMAP TRANSFER:
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
            memos = new cl_mem[]{newClt.value, newClt.grad, newClt.config};
        } else {
            memos = new cl_mem[]{newClt.value, newClt.config};
        }
        int err = clEnqueueMigrateMemObjects(
                _queue,
                memos.length,
                memos
                ,
                CL_MIGRATE_MEM_OBJECT_HOST,
                0,
                null,
                null
        );
        _mapping.put(tensor, newClt);
        tensor.add(this);
        tensor.setIsOutsourced(true);
        return this;
    }

    @Override
    public IDevice rmv(Tsr tensor) {
        cl_tsr clt = _mapping.get(tensor);
        clReleaseMemObject(clt.config);//remove translations/shapes from device!
        clReleaseMemObject(clt.value);
        _mapping.remove(tensor);
        return this;
    }

    @Override
    public IDevice overwrite64(Tsr tensor, double[] value) {//TODO: Make value an object!
        cl_tsr clt = _mapping.get(tensor);
        if(clt.fp==1){
            overwrite32(tensor, DataHelper.doubleToFloat(value));
        } else {
            //value = new double[tensor.size()];
            clEnqueueWriteBuffer(
                    _queue,
                    clt.value,
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
    public IDevice overwrite32(Tsr tensor, float[] value) {//TODO: Make value an object!
        cl_tsr clt = _mapping.get(tensor);
        if(clt.fp==1){
            clEnqueueWriteBuffer(
                    _queue,
                    clt.value,
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
    public IDevice swap(Tsr former, Tsr replacement) {
        cl_tsr clTsr = _mapping.get(former);
        _mapping.remove(former);
        _mapping.put(replacement, clTsr);
        //replacement.add(this);
        //_replacement.setIsOutsourced(true);
        //former.remove(IDevice.class);
        return this;
    }

    @Override
    public double[] value64Of(Tsr tensor, boolean grd) {
        cl_tsr clt = _mapping.get(tensor);
        if(clt.fp==1){
            return DataHelper.floatToDouble(value32Of(tensor, grd));
        } else {
            double[] data = new double[tensor.size()];
            clEnqueueReadBuffer(
                    _queue,
                    (grd)?clt.grad:clt.value,
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
    public float[] value32Of(Tsr tensor, boolean grd){
        cl_tsr clt = _mapping.get(tensor);
        if(clt.fp==1){
            float[] data = new float[tensor.size()];
            clEnqueueReadBuffer(
                    _queue,
                    (grd)?clt.grad:clt.value,
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
            return DataHelper.doubleToFloat(value64Of(tensor, grd));
        }
    }

    @Override
    public IDevice execute(Tsr[] tsrs, int f_id, int d)
    {
        if(IFunction.TYPES.REGISTER[f_id]=="<")
        {
            int offset = (tsrs[0]==null)?1:0;
            this._execute(new Tsr[]{tsrs[0+offset], tsrs[1+offset]}, IFunction.TYPES.LOOKUP.get("idy"), -1);
        }
        else if(IFunction.TYPES.REGISTER[f_id]==">")
        {
            int offset = (tsrs[0]==null)?1:0;
            _execute(new Tsr[]{tsrs[1+offset], tsrs[0+offset]}, IFunction.TYPES.LOOKUP.get("idy"), -1);
        }
        else
        {
            if(tsrs[0]==null)//Creating a new tensor:
            {
                int[] shp = (IFunction.TYPES.REGISTER[f_id] == "x")
                                ? Tsr.fcn.indexing.shpOfCon(tsrs[1].shape(), tsrs[2].shape())
                                : tsrs[1].shape();
                Tsr output = new Tsr(shp, 0.0);
                this.add(output);
                tsrs[0] = output;
            }
            if (
                    tsrs.length == 3
                            &&
                            (
                                    (tsrs[1].isVirtual() || tsrs[2].isVirtual())
                                            ||
                                    (!tsrs[1].isOutsourced() && tsrs[1].size() == 1 || !tsrs[2].isOutsourced() && tsrs[2].size() == 1)
                            )
            ) {
                if (tsrs[2].isVirtual() || tsrs[2].size() == 1) {
                    _execute(new Tsr[]{tsrs[0], tsrs[1]}, IFunction.TYPES.LOOKUP.get("idy"), -1);
                    _execute(tsrs[0], tsrs[2].value64()[0], f_id, d);
                } else {
                    _execute(new Tsr[]{tsrs[0], tsrs[2]}, IFunction.TYPES.LOOKUP.get("idy"), -1);
                    _execute(tsrs[0], tsrs[1].value64()[0], f_id, d);
                }
            } else {
                for (Tsr t : tsrs) {
                    if (!t.isOutsourced()) {
                        this.add(t);
                    }
                }
                _execute(tsrs, f_id, d);
            }
        }
        return this;
    }

    private Tsr[] _subset(Tsr[] tsrs, int padding, int start, int end){
        Tsr[] newTsrs = new Tsr[padding+end-start];
        for(int i=padding; i<newTsrs.length; i++){
            newTsrs[padding+i] = tsrs[i+start-padding];
        }
        return newTsrs;
    }

    private Tsr[] _offsetted(Tsr[] tsrs, int offset){
        Tsr[] newTsrs = new Tsr[tsrs.length-offset];
        newTsrs[0] = Tsr.fcn.create.newTsrLike(tsrs[1]);//new Tsr(tsrs[1].shape());
        if(!tsrs[1].has(GraphNode.class)){//Deleting intermediate results!
            tsrs[1].delete();
            tsrs[1] = null;
        }
        if(!tsrs[2].has(GraphNode.class)){//Deleting intermediate results!
            tsrs[2].delete();
            tsrs[2] = null;
        }
        for(int i=(1+offset); i<tsrs.length; i++){
            newTsrs[i-offset] = tsrs[i];
        }
        newTsrs[1] = tsrs[0];
        return newTsrs;
    }

    public void _execute(Tsr[] tsrs, int f_id, int d){
        for(Tsr t : tsrs){
            if(!t.isOutsourced()){
                this.add(t);
            }
        }
        if(tsrs.length>3){
            if(d<0){
                _execute(new Tsr[]{tsrs[0], tsrs[1], tsrs[2]}, f_id, d);
                Tsr[] newTsrs = _offsetted(tsrs, 1);
                _execute(newTsrs, f_id, d);//This recursion should work!
                tsrs[0] = newTsrs[0];
            } else {
                switch(Function.TYPES.REGISTER[f_id]){
                    case "+": tsrs[0] = Tsr.fcn.create.newTsrLike(tsrs[1]).setTargetValue(1.0f);
                        break;
                    case "-": tsrs[0] = Tsr.fcn.create.newTsrLike(tsrs[1]).setTargetValue((d==0)?1.0f:-1.0f);
                        break;
                    case "^":
                        if(d>0){
                            for(int i=0; i<d; i++) {

                            }
                        }
                        Tsr[] newTrs = _subset(tsrs, 1,  d+1, tsrs.length);
                        newTrs[0] =  Tsr.fcn.create.newTsrLike(tsrs[1]);
                        _execute(newTrs, IFunction.TYPES.LOOKUP.get("*"), -1);

                        break;
                    case "*":
                        break;
                    default:
                        break;
                }
            }
        } else {
            int gwz = (tsrs[0]!=null)?tsrs[0].size():tsrs[1].size();
            int offset = (tsrs[0]!=null)?0:1;
            cl_kernel kernel = _platform.getKernels().get(_platform.kernelNameOf(f_id));
            cl_mem drn = tsrs[offset].gradientIsTargeted()?_mapping.get(tsrs[offset]).grad:_mapping.get(tsrs[offset]).value;
            cl_mem src1 = tsrs[offset+1].gradientIsTargeted()?_mapping.get(tsrs[offset+1]).grad:_mapping.get(tsrs[offset+1]).value;
            if(IFunction.TYPES.isFunction(f_id)){
                clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(drn));//=> drain
                clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(_mapping.get(tsrs[offset]).config));
                clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(src1));//=>src1
                clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(_mapping.get(tsrs[offset+1]).config));
                clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{ tsrs[0].rank() }));
                clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{ d }));
            } else {
                cl_mem src2 = tsrs[offset+2].gradientIsTargeted()?_mapping.get(tsrs[offset+2]).grad:_mapping.get(tsrs[offset+2]).value;
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
    }

    public void _execute(Tsr t, double value, int f_id, int d)
    {
        if(d<0)
        {
            //execute
            int gwz = t.size();
            cl_kernel kernel = _platform.getKernels().get(
                    _platform.kernelNameOf(f_id)+"_broadcast"
            );
            cl_mem drn = t.gradientIsTargeted()?_mapping.get(t).grad:_mapping.get(t).value;
            cl_mem src1 = t.gradientIsTargeted()?_mapping.get(t).grad:_mapping.get(t).value;
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
        } else {
            /**   Derivatives implementation: (values cannot be derived)    **/
            if(
                 IFunction.TYPES.REGISTER[f_id]=="+"||
                 IFunction.TYPES.REGISTER[f_id]=="-"||
                 IFunction.TYPES.REGISTER[f_id]=="%"
            ){
                _execute(t, 0, IFunction.TYPES.LOOKUP.get("*"), -1);
                _execute(t, 1, IFunction.TYPES.LOOKUP.get("+"), -1);
            } else if(IFunction.TYPES.REGISTER[f_id]=="^"){
                _execute(t, value-1, IFunction.TYPES.LOOKUP.get("^"), -1);
                _execute(t, value, IFunction.TYPES.LOOKUP.get("*"), -1);
            } else if(IFunction.TYPES.REGISTER[f_id]=="*"){
                _execute(t, 0, IFunction.TYPES.LOOKUP.get("*"), -1);
                _execute(t, value, IFunction.TYPES.LOOKUP.get("+"), -1);
            } else if(IFunction.TYPES.REGISTER[f_id]=="/"){
                _execute(t, 0, IFunction.TYPES.LOOKUP.get("*"), -1);
                _execute(t, 1/value, IFunction.TYPES.LOOKUP.get("+"), -1);
            }
        }
    }

    //---

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
