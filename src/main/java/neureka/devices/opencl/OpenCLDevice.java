/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ____                    _____ _      _____             _
  / __ \                  / ____| |    |  __ \           (_)
 | |  | |_ __   ___ _ __ | |    | |    | |  | | _____   ___  ___ ___
 | |  | | '_ \ / _ \ '_ \| |    | |    | |  | |/ _ \ \ / / |/ __/ _ \
 | |__| | |_) |  __/ | | | |____| |____| |__| |  __/\ V /| | (_|  __/
  \____/| .__/ \___|_| |_|\_____|______|_____/ \___| \_/ |_|\___\___|
        | |
        |_|

------------------------------------------------------------------------------------------------------------------------

   'Any fool can write code that a computer can understand.
    Good programmers write code that humans can understand.'
    – Martin Fowler

    Use the following as search keys :)

    $(1) : NESTED CLASSES
    $(2) : FIELD VARIABLES
    $(3) : CONSTRUCTION
    $(4) : OPENCL PROPERTIES

*/

package neureka.devices.opencl;

import neureka.Component;
import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.AbstractDevice;
import neureka.devices.Device;
import neureka.devices.opencl.utility.CLFunctionCompiler;
import neureka.dtype.custom.F32;
import neureka.framing.Relation;
import neureka.utility.DataConverter;
import org.jocl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

import static org.jocl.CL.*;

/**
 *  This class is a concrete implementation of the {@link Device} interface by extending the {@link AbstractDevice} class.
 *  Instances of this class internally utilize the OpenCL API in order to use supported
 *  accelerator hardware like GPUs or FPGAs for storing tensors and executing operations on them.
 */
public class OpenCLDevice extends AbstractDevice<Number>
{
    private static Logger _LOG = LoggerFactory.getLogger(OpenCLDevice.class);

    public static OpenCLDevice newInstanceOf( OpenCLPlatform platform, cl_device_id did )
    {
        if ( !platform.has( did ) ) platform.put( did,  new OpenCLDevice( platform, did ) );
        return platform.get( did );
    }

    public String toString() {
        return "OpenCLDevice(deviceId=" + this._deviceId + ", platform=" + this._platform + ")";
    }

    public cl_device_id getDeviceId() {
        return this._deviceId;
    }

    public OpenCLPlatform getPlatform() {
        return this._platform;
    }

    /*==================================================================================================================
    |
    |       §(1) : NESTED CLASSES
    |   ---------------------------
    */

    /**
     * This class is an OpenCL-Device specific tensor component used to store
     * the floating point size ( 1:float, 2:double, ...),
     * a reference to a wrapper containing a pointer to the tensors configuration (cl_config),
     * and
     * a reference to a wrapper containing a pointer to the tensors data (cl_data)
     * The latter two lend their identity for garbage collection!
     */
    static class cl_tsr implements Component<Tsr<Number>>
    {

        /**
         * This class is responsible for representing the
         * data of a tensor stored on the device.
         * Instances of this class lend their identity to utilize garbage collection
         * of the data that they reference via their "cl_mem" field.
         * Meaning this inner memory object "cl_mem" will
         * be freed via a call hook stored inside a Cleaner instance...
         */
        public static class cl_value
        {
            public int size = 0;
            public cl_mem data;
            public cl_event event;
        }

        /**
         * This is the class responsible for representing NDConfiguration data.
         * Instances of this class lend their identity to utilize garbage collection
         * of the data that they reference via their "cl_mem" field.
         * Meaning this inner memory object "cl_mem" will
         * be freed via a call hook stored inside a Cleaner instance...
         */
        public static class cl_config
        {
            public cl_mem data;
        }

        public int fp = 1;
        public cl_config config = new cl_config();// Tensor configurations are always unique!
        public cl_value value;

        @Override
        public boolean update( OwnerChangeRequest<Tsr<Number>> changeRequest ) {
            // Update not needed...
            changeRequest.executeChange();
            return true;
        }
    }

    /**
     *  This class manages reference to a so called "ad hoc" program & kernel.
     *  Ad hoc is a Latin phrase meaning literally 'to this'.
     *  In English, it generally signifies a solution designed for a specific problem or task,
     *  non-generalizable, and not intended to be adapted to other purposes.
     *  This leads to the purpose of instances of this class, namely to hold the context to a unique kernel with
     *  a uniquely associated purpose which has been created by an operation possibly for specific
     *  tensor dimensions or possibly other properties...
     */
    private static class cl_ad_hoc
    {
        public String source;
        public cl_kernel kernel;
        public cl_program program;
    }

    /*==================================================================================================================
    |
    |       §(2) : FIELD VARIABLES
    |   ---------------------------
    */

    private final Map<String, cl_ad_hoc> _adhocKernels = new WeakHashMap<>();
    private final cl_ad_hoc[] _adhocKernelRingBuffer = new cl_ad_hoc[ 128 ];
    private int _ringIndex = 0;

    private final Set<Tsr<Number>> _tensors = Collections.newSetFromMap( new WeakHashMap<Tsr<Number>, Boolean>() );

    private final cl_device_id _deviceId;

    /**
     *  The OpenCLPlatform :
     *  This method is a simple getter for the OpenCLPlatform instance hosting this current device.
     *  A platform would for example be vendor specific like Intel, AMD, Nvidia...
     *
     * @return The OpenCLPlatform instance representing the platform (amd, intel, nvidia) to which this device belongs.
     */
    private final OpenCLPlatform _platform;

    /**
     * The OpenCL command queue
     */
    private final cl_command_queue _queue;


    /*==================================================================================================================
    |
    |       §(3) : CONSTRUCTION
    |   ---------------------------
    */

    /**
     * @param platform
     * @param deviceId
     */
    private OpenCLDevice( OpenCLPlatform platform, cl_device_id deviceId )
    {
        super();
        _deviceId = deviceId;
        _platform = platform;
        _queue = clCreateCommandQueueWithProperties(// Create a command-queue for the selected device
                platform.getContext(), deviceId,
                null,
                null
        );
        _cleaning( this, () -> clReleaseCommandQueue( _queue ) );
        //Runtime.getRuntime().addShutdownHook(new Thread(() -> {
        //    _mapping.forEach((k, v) -> {
        //        if (v.value.event!=null) clWaitForEvents(1, new cl_event[]{v.value.event});
        //        clReleaseMemObject(v.config);
        //        clReleaseMemObject(v.value.data);
        //    });
        //    clReleaseCommandQueue(_queue);
        //    clReleaseContext(_context);
        //}));
    }

    public boolean hasAdHocKernel( String name ) {
        return _adhocKernels.containsKey( name );
    }

    public KernelCaller getAdHocKernel( String name ) {
        cl_ad_hoc adHoc = _adhocKernels.get( name );
        if ( adHoc != null ) return new KernelCaller( adHoc.kernel, _queue );
        else return null;
    }

    /**
     *  This method compiles so called "ad hoc" kernel.
     *  Ad hoc is a Latin phrase meaning literally 'to this'.
     *  In English, it generally signifies a solution designed for a specific problem or task,
     *  non-generalizable, and not intended to be adapted to other purposes.
     *  This leads to the purpose of ad hoc kernel compilation, namely to be able to compile
     *  unique kernels with a specific purpose created on the fly during runtime by operations.
     *  This might be useful for high performance operations on tensors with specific dimensions and
     *  or possibly other variables / properties which might be taken into account...
     *
     * @param name The name of the kernel which ought to be compiled.
     * @param source The source of the kernel which ought to be compiled.
     * @return This very instance in order to enable the factory pattern.
     */
    public synchronized OpenCLDevice compileAdHocKernel( String name, String source )
    {
        if ( this.hasAdHocKernel( name ) ) {
            cl_ad_hoc adHoc = _adhocKernels.get( name );
            String message = "Cannot compile kernel source for name '"+name+"' because the name is already taken.\n" +
                    "Use another name or find out why this kernel already exists.\n" +
                    (
                            ( adHoc.source.equals( source ) )
                                    ? "Besides the name, the source code of the existing kernel is also identical.\n" : ""
                    );
            _log.error( message );
            throw new IllegalArgumentException( message );
        }

        // Create the program for the kernel
        cl_program cpProgram = clCreateProgramWithSource(
                getPlatform().getContext(),
                1,
                new String[]{source},
                null,
                null
        );

        // Build the program
        int err = clBuildProgram(
                cpProgram,
                1,
                new cl_device_id[]{ _deviceId },
                "-cl-mad-enable",
                null,
                null
        );
        //TODO: check compilation errors!
        cl_kernel kernel;
        try {
            // Create the kernel
            kernel = clCreateKernel(cpProgram, name, null);
        } catch ( Exception e ) {
            if ( e.getMessage().equals("CL_INVALID_KERNEL_NAME") && !source.contains( "__kernel void " + name ) ) {
                String message = "Method 'clCreateKernel' failed! The name of the '__kernel' method declared inside \n" +
                        "the source String does not match the provided name needed for kernel creation.";
                _log.error( message, e );
                throw new IllegalArgumentException( message );
            }
            _log.error( "Method call 'clCreateKernel(.., name=\""+name+"\", ..)' failed!", e );
            throw e;
        }
        cl_ad_hoc adHoc = new cl_ad_hoc();
        adHoc.source = source;
        adHoc.kernel = kernel;
        adHoc.program = cpProgram;
        // Storing the ad hoc object in a weak hash map for fast access by operations :
        _adhocKernels.put( name, adHoc );
        // Storing the ad hoc object in a ring buffer to avoid immediate garbage collection :
        _ringIndex = ( _ringIndex + 1 ) % _adhocKernelRingBuffer.length;

        _cleaning( adHoc, () -> {
            clReleaseKernel( kernel );
            clReleaseProgram( cpProgram );
        } );
        return this;
    }


    /**
     *  This method returns all the tensors stored on this
     *  OpenCLDevice instance as a Collection.
     *
     * @return A collection of all tensors currently stored on the device.
     */
    @Override
    public synchronized Collection<Tsr<Number>> getTensors() {
        Collection<Collection<Tsr<Number>>> collection = Collections.singleton( _tensors );
        Collection<Tsr<Number>> extracted = new ArrayList<>();
        collection.forEach( c -> c.forEach( t -> { if ( t != null ) extracted.add( t ); }));
        return extracted;
    }

    @Override
    public Operation optimizedOperationOf( Function function, String name ) {
        return new CLFunctionCompiler(this, function, name ).optimize();
    }

    /**
     *  This method tells the to restore all tensors stored on it and release all resources.
     */
    @Override
    public void dispose() {
        _tensors.forEach( this::restore );
        clFinish( _queue );
    }

    /**
     *  This method assumes that the passed tensor is stored on this device instance.
     *  If the tensor is stored on the device then the method loads the outsourced
     *  data of the tensor back into primitive JVM arrays and restores the tensor
     *  freshly in RAM.
     *
     * @param tensor The tensor whose data ought to be restored (loaded to RAM).
     * @return This device, which enables method chaining.
     */
    @Override
    public Device<Number> restore( Tsr<Number> tensor )
    {
        if ( !this.has( tensor ) ) {
            String message = "The passed tensor cannot be restored from this OpenCL device " +
                    "because the tensor is not stored on the device.\n";
            _log.error( message );
            throw new IllegalArgumentException( message );
        }
        double[] value = ( tensor.isVirtual() )
                ? _value64f( tensor.get( cl_tsr.class ), 1, 0 )
                : value64f( tensor );
        free( tensor );
        tensor.forComponent( Tsr.class, this::restore);
        tensor.setValue( value );
        return this;
    }

    @Override
    public <T extends Number> Device<Number> store( Tsr<T> tensor, Tsr<T> parent ) {
        _store( tensor, parent, () -> tensor.set( (Component) this ) );
        return this;
    }

    private <T extends Number> Device<Number> _store( Tsr<T> tensor, Tsr<T> parent, Runnable migration ) {
        if ( !parent.isOutsourced() ) throw new IllegalStateException( "Data parent is not outsourced!" );
        _add( (Tsr<Number>) tensor, parent.get( cl_tsr.class ), migration );
        _tensors.add((Tsr<Number>) tensor);
        return this;
    }

    private void _add( Tsr<Number> tensor, cl_tsr parent, Runnable migration  )
    {
        if ( this.has( tensor ) ) {
            _LOG.debug("Trying to add a tensor to a device which already reports hosting it.");
            return;
        }
        cl_tsr newClt = new cl_tsr();

        //VALUE TRANSFER:
        if ( parent == null ) {
            newClt.value = new cl_tsr.cl_value();
            _store( tensor, newClt, 1 );
            if ( tensor.rqsGradient() && tensor.has( Tsr.class ) ) this.store( tensor.getGradient() );
            {
                final cl_mem clValMem = newClt.value.data;
                cl_event clValEvent = newClt.value.event;
                _cleaning( newClt.value, () -> {
                    if ( clValEvent != null ) clWaitForEvents( 1, new cl_event[]{ clValEvent } );
                    clReleaseMemObject( clValMem );//Removing value.. from device!
                });
            }
        } else { // Tensor is a subset tensor of parent:
            newClt.fp = parent.fp;
            newClt.value = parent.value;
        }
        //CONFIG TRANSFER: <[ shape | translation | indicesMap | indices | scale ]>
        int[] config = tensor.getNDConf().asInlineArray();

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
        {
            final cl_mem clConfMem = newClt.config.data;
            _cleaning( newClt.config, () -> clReleaseMemObject( clConfMem ) );
        }
        cl_mem[] memos;
        memos = new cl_mem[]{ newClt.value.data, newClt.config.data };

        clEnqueueMigrateMemObjects(
                _queue,
                memos.length,
                memos,
                CL_MIGRATE_MEM_OBJECT_HOST,
                0,
                null,
                null
        );

        _tensors.add( tensor );

        tensor.set( newClt );
        migration.run(); // TODO: REMOVE

        if ( tensor.isVirtual() ) {
            double value = tensor.value64( 0 );
            tensor.setIsOutsourced( true );
            CalcUtil.recursiveExecution(
                    ExecutionCall.of(tensor, Tsr.of( value ).to( this ))
                                .andArgs(Arg.DerivIdx.of(-1))
                                .running(Neureka.get().context().getOperation( "<" ))
                                .on(this),
                    (executionCall, executor) -> null
            );
        }
        else tensor.setIsOutsourced( true );

        tensor.toType( F32.class );
    }

    /**
     * This method checks if the passed tensor
     * is stored on this very OpenCLDevice instance.
     * "Stored" means that the data of the tensor is represented as
     * cl_mem objects which are referenced inside tensors as components...
     *
     * @param tensor The tensor in question.
     * @return The truth value of the fact that the provided tensor is on this device.
     */
    @Override
    public <T extends Number> boolean has( Tsr<T> tensor ) { return _tensors.contains( tensor ); }


    private void _store( Tsr<Number> tensor, cl_tsr newClTsr, int fp ) {
        Pointer p;
        int size;
        //if ( !tensor.isVirtual() ) {
            if ( fp == 1 ) {
                float[] data = tensor.value32();
                data = ( data == null ) ? new float[ tensor.size() ] : data;
                p = Pointer.to(data);
                size = data.length;
            } else {
                double[] data = tensor.value64();
                data = ( data == null ) ? new double[ tensor.size() ] : data;
                p = Pointer.to(data);
                size = data.length;
            }
        //}
        newClTsr.value.size = size;
        //VALUE TRANSFER:
        cl_mem mem = clCreateBuffer(
                _platform.getContext(),
                CL_MEM_READ_WRITE,
                size * (long) Sizeof.cl_float * fp,
                null,
                null
        );
        newClTsr.value.data = mem;
        if ( !tensor.isVirtual() ) {
            clEnqueueWriteBuffer(
                    _queue,
                    mem,
                    CL_TRUE,
                    0,
                    size * (long) Sizeof.cl_float * fp,
                    p,
                    0,
                    null,
                    null
            );
        }
    }


    @Override
    public <T extends Number> Device<Number> free( Tsr<T> tensor ) {
        cl_tsr clt = tensor.get( cl_tsr.class );
        if ( clt == null ) return this;
        _tensors.remove( tensor );
        tensor.setIsOutsourced( false );
        ((Tsr<Number>)tensor).remove( cl_tsr.class );
        return this;
    }


    @Override
    public Device<Number> overwrite64( Tsr<Number> tensor, double[] value )
    {
        cl_tsr clt = tensor.get(cl_tsr.class);
        if ( clt.fp == 1 ) overwrite32( tensor, DataConverter.Utility.doubleToFloat( value ) );
        else {
            if ( clt.value.event != null ) clWaitForEvents( 1, new cl_event[]{ clt.value.event } );
            clt.value.event = new cl_event();
            clEnqueueWriteBuffer(
                    _queue,
                    clt.value.data,
                    CL_FALSE,
                    0,
                    Sizeof.cl_double * value.length,
                    Pointer.to( value ),
                    0,
                    null,
                    clt.value.event
            );
        }
        return this;
    }


    @Override
    public Device<Number> overwrite32( Tsr<Number> tensor, float[] value) {
        cl_tsr clt = tensor.get( cl_tsr.class );
        if ( clt.fp == 1 ) {
            if ( clt.value.event != null ) clWaitForEvents( 1, new cl_event[]{ clt.value.event } );
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
        }
        else overwrite64( tensor, DataConverter.Utility.floatToDouble( value ) );

        return this;
    }

    @Override
    public Device<Number> swap(Tsr<Number> former, Tsr<Number> replacement)
    {
        cl_tsr clTsr = former.get( cl_tsr.class );
        former.remove( cl_tsr.class );
        replacement.set( clTsr );
        _tensors.remove( former );
        _tensors.add( replacement );
        return this;
    }

    @Override
    public boolean update( OwnerChangeRequest<Tsr<Number>> changeRequest ) {
        super.update( changeRequest );
        if ( changeRequest.type() == IsBeing.ADDED ) {
            Tsr<Number> newOwner = changeRequest.getNewOwner();
            _updateInternal(newOwner, changeRequest::executeChange);
        }
        else
            changeRequest.executeChange();
        return true;
    }

    private void _updateInternal( Tsr newOwner, Runnable migration ) {
        Tsr<Number> root = null;
        if ( newOwner.has( Relation.class ) ) root = ((Relation<Number>)newOwner.get( Relation.class )).findRootTensor();
        if ( root != null ) store( newOwner, root );
        else _add( (Tsr<Number>) newOwner, null, migration );
    }

    public double[] value64f( Tsr<Number> tensor ) {
        cl_tsr clt = tensor.get( cl_tsr.class );
        return _value64f( clt, clt.value.size, 0 );
    }

    private double[] _value64f( cl_tsr clt , int size, int offset )
    {
        if ( clt.fp == 1 ) return DataConverter.Utility.floatToDouble( _value32f( clt, size, offset ) );
        else {
            double[] data = new double[ size ];
            clEnqueueReadBuffer(
                    _queue,
                    clt.value.data,
                    CL_TRUE,
                    offset * 8, // one double == eight byte
                    Sizeof.cl_double * data.length,
                    Pointer.to( data ),
                    0,
                    null,
                    null
            );
            return data;
        }
    }

    public float[] value32f( Tsr<Number> tensor ) {
        cl_tsr clt = tensor.get( cl_tsr.class );
        return _value32f( clt, clt.value.size, 0 );
    }

    private float[] _value32f( cl_tsr clt, int size, int offset ) {
        if ( clt.fp == 1 ) {
            float[] data = new float[ size ];
            clEnqueueReadBuffer(
                    _queue,
                    clt.value.data,
                    CL_TRUE,
                    offset * 4, // one float == four bytes !
                    (long) Sizeof.cl_float * data.length,
                    Pointer.to( data ),
                    0,
                    null,
                    null
            );
            return data;
        }
        else return DataConverter.Utility.doubleToFloat( _value64f( clt, size, offset ) );
    }

    @Override
    public Object valueFor( Tsr<Number> tensor ) {
        return value32f( tensor );
    }

    @Override
    public Number valueFor( Tsr<Number> tensor, int index ) {
        return value32f( tensor, index );
    }

    public double value64f( Tsr<Number> tensor, int index ) {
        cl_tsr clt = tensor.get( cl_tsr.class );
        return _value64f( clt, 1, index )[ 0 ];
    }

    public float value32f( Tsr<Number> tensor, int index ) {
        cl_tsr clt = tensor.get( cl_tsr.class );
        return _value32f( clt, 1, index )[ 0 ];
    }

    public KernelCaller getKernel( ExecutionCall<OpenCLDevice> call ) {
        // We create the kernel name from the chosen algorithm:
        String chosen = call.getAlgorithm().getName() + "_" + call.getOperation().getFunction();
        cl_kernel kernel = _platform.getKernels().get( chosen );
        return new KernelCaller( kernel, _queue );
    }

    @Override
    protected void _execute( Tsr[] tensors, int d, Operation type )
    {
        tensors[ 0 ].setIsVirtual( false );
    }

    /*
    // The following are two potentially important methods for future features.

    private void _releaseEvents(Tsr[] tsrs) {
        for(Tsr<Number> t : tsrs) {
            if ( t.get(cl_tsr.class).value.event != null ) {
                clReleaseEvent(t.get(cl_tsr.class).value.event);
                t.get(cl_tsr.class).value.event = null;
            }
        }
    }

    private cl_event[] _getWaitList(Tsr[] tsrs) {
        List<cl_event> list = new ArrayList<>();
        for (Tsr<Number> t : tsrs) {
            cl_event event = t.get(cl_tsr.class).value.event;
            if (event != null && !list.contains(event)) {
                list.add(event);
            }
        }
        return list.toArray(new cl_event[ 0 ]);
    }
    */


    /*==================================================================================================================
    |
    |       §(4) : OPENCL PROPERTIES
    |   ---------------------------
    */

    public String name() {
        return DeviceQuery.getString( _deviceId, CL_DEVICE_NAME );
    }

    public String vendor() {
        return DeviceQuery.getString( _deviceId, CL_DEVICE_VENDOR );
    }

    public String version() {
        return DeviceQuery.getString( _deviceId, CL_DRIVER_VERSION );
    }

    public String type() {
        long deviceType = DeviceQuery.getLong( _deviceId, CL_DEVICE_TYPE );
        if ( (deviceType & CL_DEVICE_TYPE_CPU) != 0 )
            return "CPU";
        if ( (deviceType & CL_DEVICE_TYPE_GPU) != 0 )
            return "GPU";
        if ( (deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0 )
            return "ACCELERATOR";
        if ( (deviceType & CL_DEVICE_TYPE_DEFAULT) != 0 )
            return "DEFAULT";
        return "UNKNOWN";
    }

    public int maxComputeUnits() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    public long maxWorkItemSimensions() {
        return DeviceQuery.getLong(_deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    }

    public long[] maxWorkItemSizes() {
        return DeviceQuery.getSizes(_deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
    }

    public long maxWorkGroupSize() {
        return DeviceQuery.getSize(_deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    public long maxClockFrequenzy() {
        return DeviceQuery.getLong(_deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    public int maxAddressBits() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_ADDRESS_BITS);
    }

    public long maxMemAllocSize() {
        return DeviceQuery.getLong(_deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    }

    public long globalMemSize() {
        return DeviceQuery.getLong(_deviceId, CL_DEVICE_GLOBAL_MEM_SIZE);
    }

    public int errorCorrectionSupport() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
    }

    public int localMemType() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_LOCAL_MEM_TYPE);
    }

    public long localMemSize() {
        return DeviceQuery.getLong(_deviceId, CL_DEVICE_LOCAL_MEM_SIZE);
    }

    public long maxConstantBufferSize() {
        return DeviceQuery.getLong(_deviceId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    }

    public long maxConstantBufferSizeKB() {
        return (int) (DeviceQuery.getLong(_deviceId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) / 1024);
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
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_IMAGE_SUPPORT);
    }

    public int maxReadImageArgs() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_MAX_READ_IMAGE_ARGS);
    }

    public int maxWriteImageArgs() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
    }

    public long singleFPConfig() {
        return DeviceQuery.getLong(_deviceId, CL_DEVICE_SINGLE_FP_CONFIG);
    }

    public long image2DMaxWidth() {
        return DeviceQuery.getSize(_deviceId, CL_DEVICE_IMAGE2D_MAX_WIDTH);
    }

    public long image2DMaxHeight() {
        return DeviceQuery.getSize(_deviceId, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
    }

    public long image3DMaxWidth() {
        return DeviceQuery.getSize(_deviceId, CL_DEVICE_IMAGE3D_MAX_WIDTH);
    }

    public long image3DMaxHeight() {
        return DeviceQuery.getSize(_deviceId, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
    }

    public long image3DMaxDepth() {
        return DeviceQuery.getSize(_deviceId, CL_DEVICE_IMAGE3D_MAX_DEPTH);
    }

    public int prefVecWidthChar() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
    }

    public int prefVecWidthShort() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
    }

    public int prefVecWidthInt() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
    }

    public int prefVecWidthLong() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
    }

    public int prefVecWidthFloat() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
    }

    public int prefVecWidthDouble() {
        return DeviceQuery.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
    }

    public static class DeviceQuery
    {
        /**
         *  Returns the value of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value
         */
        public static int getInt(cl_device_id device, int paramName) {
            return getInts(device, paramName, 1)[ 0 ];
        }

        /**
         *  Returns the values of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value
         */
        public static int[] getInts(cl_device_id device, int paramName, int numValues) {
            int[] values = new int[numValues];
            clGetDeviceInfo(device, paramName, (long)Sizeof.cl_int * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         *  Returns the value of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value
         */
        public static long getLong(cl_device_id device, int paramName) {
            return getLongs(device, paramName, 1)[ 0 ];
        }

        /**
         *  Returns the values of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value
         */
        public static long[] getLongs(cl_device_id device, int paramName, int numValues) {
            long[] values = new long[numValues];
            clGetDeviceInfo(device, paramName, (long)Sizeof.cl_long * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         *  Returns the value of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value
         */
        public static String getString(cl_device_id device, int paramName) {
            // Obtain the length of the string that will be queried
            long[] size = new long[ 1 ];
            clGetDeviceInfo(device, paramName, 0, null, size);

            // Create a buffer of the appropriate size and fill it with the info
            byte[] buffer = new byte[(int) size[ 0 ]];
            clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

            // Create a string from the buffer (excluding the trailing \0 byte)
            return new String(buffer, 0, buffer.length - 1);
        }

        /**
         *  Returns the value of the platform info parameter with the given name
         *
         * @param platform  The platform
         * @param paramName The parameter name
         * @return The value
         */
        public static String getString( cl_platform_id platform, int paramName ) {
            // Obtain the length of the string that will be queried
            long[] size = new long[ 1 ];
            clGetPlatformInfo(platform, paramName, 0, null, size);

            // Create a buffer of the appropriate size and fill it with the info
            byte[] buffer = new byte[ (int) size[ 0 ] ];
            clGetPlatformInfo( platform, paramName, buffer.length, Pointer.to(buffer), null );

            // Create a string from the buffer (excluding the trailing \0 byte)
            return new String( buffer, 0, buffer.length - 1 );
        }

        /**
         *  Returns the value of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value64
         */
        public static long getSize( cl_device_id device, int paramName ) {
            return getSizes( device, paramName, 1 )[ 0 ];
        }

        /**
         *  Returns the values of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value64
         */
        public static long[] getSizes(cl_device_id device, int paramName, int numValues) {
            // The size of the returned data has to depend on
            // the size of a size_t, which is handled here
            ByteBuffer buffer = ByteBuffer.allocate( numValues * Sizeof.size_t ).order( ByteOrder.nativeOrder() );
            clGetDeviceInfo(
                    device,
                    paramName,
                    (long) Sizeof.size_t * numValues,
                    Pointer.to(buffer),
                    null
            );
            long[] values = new long[ numValues ];
            if ( Sizeof.size_t == 4 ) {
                for ( int i = 0; i < numValues; i++ ) {
                    values[ i ] = buffer.getInt( i * Sizeof.size_t );
                }
            } else {
                for ( int i = 0; i < numValues; i++ ) {
                    values[ i ] = buffer.getLong( i * Sizeof.size_t );
                }
            }
            return values;
        }

    }


}
