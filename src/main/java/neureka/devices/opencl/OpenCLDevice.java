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

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.*;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.ocl.CLBackend;
import neureka.common.composition.Component;
import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.devices.AbstractDevice;
import neureka.devices.Device;
import neureka.devices.opencl.utility.CLFunctionCompiler;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.F32;
import neureka.framing.Relation;
import neureka.math.Function;
import neureka.ndim.config.NDConfiguration;
import org.jocl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.function.Supplier;

import static org.jocl.CL.*;

/**
 * This class models OpenCL supporting accelerator hardware like GPUs or FPGAs
 * for storing tensors and executing operations on them.
 */
public class OpenCLDevice extends AbstractDevice<Number>
{
    private static final Logger _LOG = LoggerFactory.getLogger(OpenCLDevice.class);

    static OpenCLDevice of( OpenCLPlatform platform, cl_device_id did ) {
        if (!platform.has(did)) platform.put(did, new OpenCLDevice(platform, did));
        return platform.get(did);
    }

    public enum Type {
        CPU, GPU, ACCELERATOR, DEFAULT, CUSTOM, ALL, UNKNOWN
    }

    enum cl_dtype { F32, F64, I64, I32, I16, I8, U32, U16, U8 }

    /*==================================================================================================================
    |
    |       §(1) : NESTED CLASSES
    |   ---------------------------
    */

    /**
     * This class is an OpenCL-Device specific tensor component used to store
     * the floating point size ( 1:float, 2:double, ...),
     * a reference to a wrapper containing a pointer to the tensor's configuration (cl_config),
     * and
     * a reference to a wrapper containing a pointer to the tensor's data (cl_data)
     * The latter two lend their identity for garbage collection!
     */
    static class cl_tsr<V, T extends V> {

        cl_tsr(cl_tsr.cl_value value, cl_dtype  dtype, cl_config config) {
            this.value = value;
            this.dtype = dtype;
            this.config = config;
        }

        /**
         * This class is responsible for representing the
         * data of a tensor stored on the device.
         * Instances of this class lend their identity to utilize garbage collection
         * of the data that they reference via their "cl_mem" field.
         * Meaning this inner memory object "cl_mem" will
         * be freed via a call hook stored inside a Cleaner instance...
         */
        static class cl_value
        {
            cl_value( int size ) { this.size = size; }

            public final int size;
            public cl_mem    data;
            public cl_event  event;
        }

        /**
         * This is the class responsible for representing NDConfiguration data.
         * Instances of this class lend their identity to utilize garbage collection
         * of the data that they reference via their "cl_mem" field.
         * Meaning this inner memory object "cl_mem" will
         * be freed via a call hook stored inside a Cleaner instance...
         */
        static final class cl_config {
            public cl_mem data;
        }

        public cl_config config;
        public final cl_dtype  dtype;
        public final cl_value  value;

        @Override
        public boolean equals(Object obj) {
            if ( !(obj instanceof cl_tsr) ) return false;
            return ((cl_tsr) obj).value == this.value;
        }
    }

    /**
     * This class manages a reference to a so called "ad hoc" program & kernel.
     * Ad hoc is a Latin phrase meaning literally 'to this'.
     * In English, it generally signifies a solution designed for a specific problem or task,
     * non-generalizable, and not intended to be adapted to other purposes.
     * This leads to the purpose of this class, namely to hold the context to a unique kernel with
     * a uniquely associated purpose which has been created by an operation possibly for specific
     * tensor dimensions or possibly other properties...
     */
    static final class cl_ad_hoc
    {
        public final String source;
        public final cl_kernel kernel;
        public final cl_program program;

        public cl_ad_hoc(
            String source, cl_kernel kernel, cl_program program
        ) {
            this.source = source;
            this.kernel = kernel;
            this.program = program;
        }
    }

    /*==================================================================================================================
    |
    |       §(2) : FIELD VARIABLES
    |   ---------------------------
    */

    private final Set<Tsr<Number>> _tensors = Collections.newSetFromMap(new WeakHashMap<>());

    private final KernelCache _kernelCache = new KernelCache();

    private final cl_device_id _deviceId;

    /**
     * The OpenCLPlatform :
     * This method is a simple getter for the OpenCLPlatform instance hosting this current device.
     * A platform would for example be vendor specific like Intel, AMD, Nvidia...
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
     * @param platform The platform containing this device.
     * @param deviceId The underlying OpenCL id of this device.
     */
    private OpenCLDevice( OpenCLPlatform platform, cl_device_id deviceId ) {
        super();
        _deviceId = deviceId;
        _platform = platform;
        _queue = clCreateCommandQueueWithProperties(// Create a command-queue for the selected device
                        platform.getContext(), deviceId,
                        null,
                        null
                    );
        _cleaning(this, () -> clReleaseCommandQueue(_queue));
    }

    public final String toString() {
        return "OpenCLDevice[id=0x" + Long.toHexString(_deviceId.getNativePointer()) + ",platform=0x" + Long.toHexString(_platform.getId()) + "]";
    }

    public final cl_device_id getId() { return _deviceId; }

    public final OpenCLPlatform getPlatform() { return _platform; }

    /**
     * @param name The name of the kernel whose presents should be checked.
     * @return True if the kernel is present in the cache, false otherwise.
     */
    public boolean hasAdHocKernel( String name ) { return _kernelCache.has(name); }

    /**
     * @param name The name of the kernel which should be retrieved.
     * @return The kernel with the given name if it is present in the cache, throws an exception otherwise.
     */
    public KernelCaller getAdHocKernel( String name ) {
        cl_ad_hoc adHoc = _kernelCache.get(name);
        if (adHoc != null) return new KernelCaller(adHoc.kernel, _queue);
        else throw new IllegalArgumentException("No ad hoc kernel with name '" + name + "' found!");
    }

    /**
     * @param name The name of the kernel which should be retrieved.
     * @return An {@link Optional} containing the kernel with the given name if it is present in the cache, an empty optional otherwise.
     */
    public Optional<KernelCaller> findAdHocKernel( String name ) {
        cl_ad_hoc adHoc = _kernelCache.get(name);
        if (adHoc != null) return Optional.of(new KernelCaller(adHoc.kernel, _queue));
        else return Optional.empty();
    }

    /**
     * @param name The name of the kernel which should be retrieved.
     * @param source The source code of the kernel which should be compiled if it is not present in the cache.
     * @return The kernel caller for the kernel of the requested name, either from cache,
     *          or compiled from the given source code if it was not present in the cache.
     */
    public KernelCaller findOrCompileAdHocKernel( String name, Supplier<String> source ) {
        cl_ad_hoc adHoc = _kernelCache.get(name);
        if ( adHoc != null ) return new KernelCaller(adHoc.kernel, _queue);
        else return compileAndGetAdHocKernel(name, source.get());
    }

    /**
     * This method compiles and returns the {@link KernelCaller} for a so called "ad hoc" kernel.
     * Ad hoc is a Latin phrase meaning literally 'to this'.
     * In English, it generally signifies a solution designed for a specific problem or task,
     * non-generalizable, and not intended to be adapted to other purposes.
     * This leads to the purpose of ad hoc kernel compilation, namely to be able to compile
     * unique kernels with a specific purpose created on the fly during runtime by operations.
     * This might be useful for high performance operations on tensors with specific dimensions and
     * or possibly other variables / properties which might be taken into account...
     *
     * @param name   The name of the kernel which ought to be compiled.
     * @param source The source of the kernel which ought to be compiled.
     * @return The {@link KernelCaller} for the compiled kernel.
     */
    public synchronized KernelCaller compileAndGetAdHocKernel( String name, String source ) {
        return compileAdHocKernel( name, source )
                .findAdHocKernel( name )
                .orElseThrow(() -> new RuntimeException("Failed to compile kernel: " + name));
    }

    /**
     * This method compiles so called "ad hoc" kernel.
     * Ad hoc is a Latin phrase meaning literally 'to this'.
     * In English, it generally signifies a solution designed for a specific problem or task,
     * non-generalizable, and not intended to be adapted to other purposes.
     * This leads to the purpose of ad hoc kernel compilation, namely to be able to compile
     * unique kernels with a specific purpose created on the fly during runtime by operations.
     * This might be useful for high performance operations on tensors with specific dimensions and
     * or possibly other variables / properties which might be taken into account...
     *
     * @param name   The name of the kernel which ought to be compiled.
     * @param source The source of the kernel which ought to be compiled.
     * @return This very instance in order to enable the factory pattern.
     */
    public synchronized OpenCLDevice compileAdHocKernel( String name, String source ) {
        if (this.hasAdHocKernel(name)) {
            cl_ad_hoc adHoc = _kernelCache.get(name);
            String message =
                "Cannot compile kernel source for name '" + name + "' because the name is already taken.\n" +
                "Use another name or find out why this kernel already exists.\n" +
                (
                        adHoc.source.equals(source)
                                ? "Besides the name, the source code of the existing kernel is also identical.\n" : ""
                );
            _log.error(message);
            throw new IllegalArgumentException(message);
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
                        new cl_device_id[]{_deviceId},
                        "-cl-mad-enable",
                        null,
                        null
                );

        if ( err != CL_SUCCESS )
            _log.error("Error when trying to compile 'ad hoc kernel' named '"+name+"'! Error code: "+err);

        //TODO: check compilation errors!
        cl_kernel kernel;
        try {
            // Create the kernel
            kernel = clCreateKernel(cpProgram, name, null);
        } catch (Exception e) {
            if (e.getMessage().equals("CL_INVALID_KERNEL_NAME") && !source.contains("__kernel void " + name)) {
                String message = "Method 'clCreateKernel' failed! The name of the '__kernel' method declared inside \n" +
                                 "the source String does not match the provided name needed for kernel creation.";
                _log.error(message, e);
                throw new IllegalArgumentException(message);
            }
            _log.error("Method call 'clCreateKernel(.., name=\"" + name + "\", ..)' failed!", e);
            throw e;
        }
        cl_ad_hoc adHoc = new cl_ad_hoc(source, kernel, cpProgram);

        // Storing the ad hoc object in a weak hash map for fast access by operations :
        _kernelCache.put( name, adHoc );

        _cleaning(adHoc, () -> {
            clReleaseKernel(kernel);
            clReleaseProgram(cpProgram);
        });
        return this;
    }


    /**
     * This method returns all the tensors stored on this
     * OpenCLDevice instance as a Collection.
     *
     * @return A collection of all tensors currently stored on the device.
     */
    @Override
    public synchronized Collection<Tsr<Number>> getTensors() {
        Collection<Collection<Tsr<Number>>> collection = Collections.singleton(_tensors);
        Collection<Tsr<Number>> extracted = new ArrayList<>();
        collection.forEach( c -> c.forEach( t -> { if ( t != null ) extracted.add(t); } ) );
        return extracted;
    }

    @Override
    public Operation optimizedOperationOf( Function function, String name ) {
        return new CLFunctionCompiler( this, function, name ).optimize();
    }

    /**
     * This method tells the to restore all tensors stored on it and release all resources.
     */
    @Override
    public void dispose() {
        _tensors.forEach( this::restore );
        clFinish( _queue );
        clReleaseCommandQueue( _queue );
    }

    /**
     * This method assumes that the passed tensor is stored on this device instance.
     * If the tensor is stored on the device then the method loads the outsourced
     * data of the tensor back into primitive JVM arrays and restores the tensor
     * freshly in RAM.
     *
     * @param tensor The tensor whose data ought to be restored (loaded to RAM).
     * @return This device, which enables method chaining.
     */
    @Override
    public Device<Number> restore( Tsr<Number> tensor ) {
        if ( !this.has( tensor ) ) {
            String message = "The passed tensor cannot be restored from this OpenCL device " +
                                "because the tensor is not stored on the device.\n";
            _log.error(message);
            throw new IllegalArgumentException(message);
        }

        Object value  = _read(JVMData.of(tensor.itemType(), tensor.isVirtual() ? 1 : tensor.size()), tensor, 0).getArray();

        Class<?> arrayType = Objects.requireNonNull(tensor.getDataType().getTypeClassInstance(NumericType.class)).holderArrayType();

        value = DataConverter.get().convert( value, arrayType );

        this.free( tensor );
        tensor.find( Tsr.class ).ifPresent( this::restore );
        tensor.getMut().setItems( value );
        return this;
    }


    /**
     * Implementations of this method ought to store the value
     * of the given tensor and the "parent" tensor in whatever
     * formant suites the underlying implementation and or final type.
     * {@link Device} implementations are also tensor storages
     * which may also have to store tensors which are slices of bigger tensors.   <br><br>
     *
     * @param tensor The tensor whose data ought to be stored.
     */
    private <T extends Number> void _store( Tsr<T> tensor, Tsr<T> parent ) {
        if (!parent.isOutsourced()) throw new IllegalStateException("Data parent is not outsourced!");
        _add(
            tensor.getMut().upcast(Number.class),
            parent.getMut().getData().getRef( cl_tsr.class ),
            () -> tensor.set((Component) this)
        );
    }

    @Override
    public final <T extends Number> boolean has( Tsr<T> tensor ) {
        return _tensors.contains(tensor);
    }

    private <T extends Number> void _add(
            Tsr<Number> tensor,
            cl_tsr<Number, T> parent,
            Runnable migration // Causes the device to be a component of the tensor!
    ) {
        if ( this.has( tensor ) ) {
            _LOG.debug("Trying to add a tensor to a device which already reports hosting it.");
            return;
        }
        if ( parent == null ) {
            if ( tensor.getMut().getData().owner() == this ) {
                _tensors.add( tensor );
                migration.run();
                return;
            }
        }

        boolean convertToFloat = Neureka.get()
                                        .backend()
                                        .find(CLBackend.class)
                                        .map( it -> it.getSettings().isAutoConvertToFloat() )
                                        .orElse(false);
        JVMData jvmData = null;

        if ( parent == null )
            jvmData = JVMData.of( tensor.getMut().getData().getRef(), convertToFloat );

        cl_tsr<Number, Number> newClt;

        if ( parent != null )
            newClt = _storeFromParent( tensor, parent );
        else {
            newClt = _storeNew( tensor.getNDConf(), jvmData );
            if ( tensor.rqsGradient() && tensor.hasGradient() )
                this.store(tensor.gradient().orElseThrow(()->new IllegalStateException("Gradient missing!")));
        }

        cl_mem[] memos = parent == null
                                ? new cl_mem[]{newClt.value.data, newClt.config.data}
                                : new cl_mem[]{newClt.config.data};

        clEnqueueMigrateMemObjects(
                _queue, memos.length, memos,
                CL_MIGRATE_MEM_OBJECT_HOST,
                0,
                null,
                null
            );

        neureka.Data<Number> data = _dataArrayOf(newClt, (DataType<Number>) _dataTypeOf(newClt));

        _tensors.add( tensor );

        tensor.getMut().setData( data );
        migration.run();

        // When tensors get stored on this device,
        // they are implicitly converted to a float tensor:
        if ( convertToFloat )
            tensor.getMut().toType(F32.class);
    }

    private cl_tsr<Number, Number> _storeFromParent( Tsr<Number> tensor, cl_tsr<Number, ?> parent ) {
        cl_tsr.cl_config config = _writeNDConfig( tensor.getNDConf() );
        return new cl_tsr<>(parent.value, parent.dtype, config);
    }

    private cl_tsr<Number, Number> _storeNew( NDConfiguration ndc, JVMData jvmData ) {
        return _storeNew( ndc, jvmData, false );
    }

    private cl_tsr<Number, Number> _storeNew( NDConfiguration ndc, JVMData jvmData, boolean allocateTargetSize ) {
        cl_tsr.cl_config config = _writeNDConfig( ndc );
        cl_tsr.cl_value newVal = new cl_tsr.cl_value((int) (allocateTargetSize ? jvmData.getTargetLength() : jvmData.getLength()));
        cl_tsr<Number, Number> newClt = new cl_tsr<>(newVal, jvmData.getType(), config);
        _store( jvmData, newClt, allocateTargetSize );
        return newClt;
    }

    private cl_tsr.cl_config _writeNDConfig( NDConfiguration ndc ) {

        cl_tsr.cl_config clf = new cl_tsr.cl_config();

        //CONFIG TRANSFER: <[ shape | translation | indicesMap | indices | scale ]>
        int[] config = ndc.asInlineArray();

        //SHAPE/TRANSLATION/IDXMAP/OFFSET/SPREAD TRANSFER:
        clf.data = clCreateBuffer(
                _platform.getContext(),
                CL_MEM_READ_WRITE,
                (long) config.length * Sizeof.cl_int,
                null, null
        );

        clEnqueueWriteBuffer(
                _queue, clf.data, CL_TRUE, 0,
                (long) config.length * Sizeof.cl_int,
                Pointer.to(config),
                0,
                null, null
        );
        final cl_mem clConfMem = clf.data;
        _cleaning(clf, () -> clReleaseMemObject(clConfMem));
        return clf;
    }

    private void _store(
       JVMData jvmData,
       cl_tsr<?, ?> newClTsr,
       boolean allocateTarget
    ) {
        long bufferLength = allocateTarget ? jvmData.getTargetLength() : jvmData.getLength();

        cl_mem mem = clCreateBuffer(
                        _platform.getContext(),
                        CL_MEM_READ_WRITE,
                        (long) jvmData.getItemSize() * bufferLength,
                        null,
                        null
                    );

        newClTsr.value.data = mem;

        // Virtual means that there is only a single value in the JVM array.
        // So we don't have to write the whole array to the device!
        // Instead, we can just fill the device memory with the single value.
        boolean isASingleValue = jvmData.isVirtual();

        if ( isASingleValue )
            clEnqueueFillBuffer(
                    _queue, mem, jvmData.getPointer(), // pattern
                    jvmData.getItemSize(), 0,
                    (long) jvmData.getItemSize() * bufferLength,
                    0, null, null
                );
        else
            clEnqueueWriteBuffer(
                    _queue, mem,
                    CL_TRUE, 0,
                    (long) jvmData.getItemSize() * bufferLength,
                    jvmData.getPointer(), 0, null, null
                );

        final cl_mem clValMem = newClTsr.value.data;
        cl_event clValEvent = newClTsr.value.event;
        _cleaning( newClTsr.value, () -> {
            if ( clValEvent != null ) clWaitForEvents(1, new cl_event[]{clValEvent});
            clReleaseMemObject(clValMem); // Removing data from the device!
        });
    }

    @Override
    public final <T extends Number> Device<Number> free( Tsr<T> tensor ) {
        cl_tsr<?, ?> clt = tensor.getMut().getData().getRef( cl_tsr.class);
        if ( clt == null ) return this;
        _tensors.remove(tensor);
        tensor.getMut().setData(null);
        tensor.find(Device.class).ifPresent(
            device -> {
                tensor.remove( Device.class );
                tensor.find(Tsr.class).ifPresent(
                    gradient ->
                        ( (Tsr<Number>) gradient ).find(Device.class).ifPresent(
                            gradDevice -> {
                                try {
                                    if ( _tensors.contains( gradient ) ) gradDevice.restore( gradient );
                                }
                                catch ( Exception exception ) {
                                    _LOG.error(
                                        "Gradient could not be restored from device component when trying to migrate it back to RAM.",
                                        exception
                                    );
                                    throw exception;
                                }
                                gradient.remove( Device.class );
                            })
                );
            }
        );
        return this;
    }

    @Override
    protected final <T extends Number> T _readItem( Tsr<T> tensor, int index ) {
        return (T) _read(JVMData.of(tensor.itemType(), 1), tensor.getMut().upcast(Number.class), index).getElementAt(0);
    }

    @Override
    protected final <T extends Number, A> A _readArray( Tsr<T> tensor, Class<A> arrayType, int start, int size ) {
        return (A) _read(JVMData.of(tensor.itemType(), size), tensor.getMut().upcast(Number.class), start).getArray();
    }

    @Override
    protected final <T extends Number> void _writeItem( Tsr<T> tensor, T item, int start, int size ) {
        _overwrite( tensor, start, JVMData.of(item, size, 0) );
    }

    @Override
    protected final <T extends Number> void _writeArray( Tsr<T> tensor, Object array, int offset, int start, int size ) {
        _overwrite( tensor, start, JVMData.of(array, size, offset) );
    }

    @Override
    public <T extends Number>  neureka.Data<T> allocate( DataType<T> dataType, NDConfiguration ndc ) {
        JVMData jvmData = JVMData.of( dataType.getItemTypeClass(), ndc.size() );
        cl_tsr<Number, Number> clt = _storeNew( ndc, jvmData );
        return (neureka.Data<T>) _dataArrayOf(clt, (DataType<Number>) _dataTypeOf(clt));
    }

    @Override
    public <T extends Number>  neureka.Data<T> allocateFromOne( DataType<T> dataType, NDConfiguration ndc, T initialValue ) {
        JVMData jvmData = JVMData.of( initialValue, ndc.size(), false, true );
        cl_tsr<Number, Number> clt = _storeNew( ndc, jvmData );
        return (neureka.Data<T>) _dataArrayOf(clt, (DataType<Number>) _dataTypeOf(clt));
    }

    @Override
    public <T extends Number> neureka.Data<T> allocateFromAll( DataType<T> dataType, NDConfiguration ndc, Object data ) {
        JVMData jvmData = JVMData.of( data );
        cl_tsr<Number, Number> clt = _storeNew( ndc, jvmData );
        return (neureka.Data<T>) _dataArrayOf(clt, (DataType<Number>) _dataTypeOf(clt));
    }

    @Override
    protected neureka.Data<Number> _actualize( Tsr<?> tensor ) {
        NDConfiguration ndc = tensor.getNDConf();
        Object initialValue = tensor.item();
        cl_tsr<?, ?> clt = tensor.getMut().getData().getRef( cl_tsr.class);
        if ( clt == null ) throw new IllegalStateException("The tensor has no device component!");
        JVMData jvmData = JVMData.of( initialValue, ndc.size(), false, true );
        clt = _storeNew( ndc, jvmData, true );
        return _dataArrayOf(clt, (DataType<Number>) _dataTypeOf(clt));
    }

    @Override
    protected neureka.Data<Number> _virtualize( Tsr<?> tensor ) {
        NDConfiguration ndc = tensor.getNDConf();
        Object initialValue = tensor.item();
        cl_tsr<?, ?> clt = tensor.getMut().getData().getRef( cl_tsr.class);
        if ( clt == null ) throw new IllegalStateException("The tensor has no device component!");
        JVMData jvmData = JVMData.of( initialValue, ndc.size(), false, true );
        clt = _storeNew( ndc, jvmData, false );
        return _dataArrayOf(clt, (DataType<Number>) _dataTypeOf(clt));
    }

    @Override
    protected final DataType<?> _dataTypeOf( Object rawData ) {
        LogUtil.nullArgCheck( rawData, "rawData", Object.class );
        if ( rawData instanceof cl_tsr ) {
            cl_dtype type = ((cl_tsr) rawData).dtype;
            switch ( type ) {
                case F32: return DataType.of( Float.class );
                case F64: return DataType.of( Double.class );
                case I32: case U32:
                    return DataType.of( Integer.class );
                case I64: return DataType.of( Long.class );
                case I16: case U16:
                    return DataType.of( Short.class );
                case I8: case U8:
                    return DataType.of( Byte.class );
                default: throw new IllegalStateException("Unknown OpenCL data type!");
            }
        }
        throw new IllegalStateException("Unknown data type "+rawData.getClass()+"!");
    }

    private void _overwrite(
            Tsr<?> tensor, long offset, JVMData jvmData
    ) {
        if ( jvmData.getLength() == 0 ) return;
        cl_tsr<?, ?> clt = tensor.getMut().getData().getRef( cl_tsr.class);

        if ( clt.value.event != null ) clWaitForEvents(1, new cl_event[]{clt.value.event});
        clt.value.event = new cl_event();
        long start = offset * jvmData.getItemSize();
        long size  = jvmData.getItemSize() * jvmData.getLength();
        clEnqueueWriteBuffer(
                _queue, clt.value.data, CL_TRUE,
                start, size,
                jvmData.getPointer(), 0, null,
                clt.value.event
            );
    }

    @Override
    protected final <T extends Number> void _swap(Tsr<T> former, Tsr<T> replacement) {
        cl_tsr<Number, T> clTsr = former.getMut().getData().getRef( cl_tsr.class);
        former.getMut().setData(null);
        replacement.getMut().setData( _dataArrayOf(clTsr, (DataType<T>) _dataTypeOf(clTsr)) );
        _tensors.remove(former);
        _tensors.add( replacement.getMut().upcast(Number.class) );
    }

    @Override
    public boolean update( OwnerChangeRequest<Tsr<Number>> changeRequest ) {
        super.update(changeRequest);
        if (changeRequest.type() == IsBeing.ADDED) {
            Tsr<Number> newOwner = changeRequest.getNewOwner();
            _updateInternal(newOwner, changeRequest::executeChange);
        } else
            changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
        return true;
    }

    @Override
    protected <T extends Number> void _updateNDConf( Tsr<T> tensor ) {
        cl_tsr<?, ?> clt = tensor.getMut().getData().getRef( cl_tsr.class);
        if ( clt != null ) {
            // This will create a new cl config.
            clt.config = _writeNDConfig(tensor.getNDConf());
            // The old one will be garbage collected through the cleaner!
            cl_mem[] memos = new cl_mem[]{clt.config.data};
            clEnqueueMigrateMemObjects(
                    _queue,
                    memos.length,
                    memos,
                    CL_MIGRATE_MEM_OBJECT_HOST,
                    0,
                    null,
                    null
            );
        }
    }

    @Override
    protected <T extends Number> int _sizeOccupiedBy( Tsr<T> tensor ) { return tensor.getMut().getData().getRef( cl_tsr.class).value.size; }

    @Override
    protected <T extends Number> Object _readAll( Tsr<T> tensor, boolean clone ) {
        cl_tsr<?, ?> clt = tensor.getMut().getData().getRef( cl_tsr.class);
        return _readArray( tensor, tensor.getDataType().dataArrayType(), 0, clt.value.size );
    }

    private void _updateInternal(Tsr<Number> newOwner, Runnable migration) {
        Tsr<Number> root = _findRoot( newOwner );
        if (root != null) _store(newOwner, root);
        else _add( newOwner, null, migration );
    }

    private Tsr<Number> _findRoot( Tsr<Number> newOwner ) {
        Tsr<Number> root = null;
        Relation<Number> relation = newOwner.get(Relation.class);
        if ( relation != null )
            root = ((Relation<Number>) newOwner.get(Relation.class)).findRootTensor().orElse(null);

        return root;
    }

    private JVMData _read( JVMData jvmData, Tsr<Number> tensor, int offset ) {
        cl_tsr<?, ?> clt = tensor.getMut().getData().getRef( cl_tsr.class);
        clEnqueueReadBuffer(
                _queue,
                clt.value.data,
                CL_TRUE,
                (long) offset * jvmData.getItemSize(), // one double == eight byte
                (long) jvmData.getItemSize() * jvmData.getLength(),
                jvmData.getPointer(),
                0,
                null,
                null
        );
        return jvmData;
    }

    /**
     * @param call The {@link ExecutionCall} which will be queried for a {@link CLImplementation} holding the kernel.
     * @return The kernel call which uses the builder pattern to receive kernel arguments.
     */
    public KernelCaller getKernel( ExecutionCall<OpenCLDevice> call ) {
        String chosen;
        Algorithm algorithm = call.getAlgorithm();
        DeviceAlgorithm<?> deviceAlgorithm = ( algorithm instanceof DeviceAlgorithm ? ((DeviceAlgorithm<?>) algorithm) : null );
        // We create the kernel name from the chosen algorithm:
        ImplementationFor<OpenCLDevice> impl = ( deviceAlgorithm == null ? null : deviceAlgorithm.getImplementationFor(OpenCLDevice.class) );
        if ( impl instanceof CLImplementation && _platform.hasKernel(((CLImplementation) impl).getKernelFor(call).getName()) ) {
            chosen = ((CLImplementation) impl).getKernelFor( call ).getName();
        }
        else
            chosen = call.getAlgorithm().getName() + "_" + call.getOperation().getIdentifier();

        cl_kernel kernel = _platform.getKernel( chosen );
        if ( kernel == null )
            throw new IllegalStateException(
                    "No kernel found for signature '" + chosen + "' for operation '" +  call.getOperation().getIdentifier() + "'."
                );

        return new KernelCaller(kernel, _queue);
    }

    /**
     * @param name The name of the kernel for which a {@link KernelCaller} should be returned.
     * @return A {@link KernelCaller} for calling the requested kernel.
     */
    public KernelCaller getKernel( String name ) {
        cl_kernel kernel = _platform.getKernel( name );
        if ( kernel == null )
            throw new IllegalStateException("No kernel found with name '" + name + "'.");
        return new KernelCaller(kernel, _queue);
    }

    @Override
    protected boolean _approveExecutionOf( Tsr<?>[] tensors, int d, Operation type ) { return true; }


    /*==================================================================================================================
    |
    |       §(4) : OPENCL PROPERTIES
    |   ---------------------------
    */

    public String name() { return Query.getString( _deviceId, CL_DEVICE_NAME ); }

    public String vendor() { return Query.getString(_deviceId, CL_DEVICE_VENDOR); }

    public String version() { return Query.getString(_deviceId, CL_DRIVER_VERSION); }

    public Type type() {
        long deviceType = Query.getLong(_deviceId, CL_DEVICE_TYPE);
        if ( (deviceType & CL_DEVICE_TYPE_CPU         ) != 0 ) return Type.CPU;
        if ( (deviceType & CL_DEVICE_TYPE_GPU         ) != 0 ) return Type.GPU;
        if ( (deviceType & CL_DEVICE_TYPE_ACCELERATOR ) != 0 ) return Type.ACCELERATOR;
        if ( (deviceType & CL_DEVICE_TYPE_DEFAULT     ) != 0 ) return Type.DEFAULT;
        if ( (deviceType & CL_DEVICE_TYPE_CUSTOM      ) != 0 ) return Type.CUSTOM;
        if ( (deviceType & CL_DEVICE_TYPE_ALL         ) != 0 ) return Type.ALL;
        return Type.UNKNOWN;
    }

    public int maxComputeUnits() { return Query.getInt(_deviceId, CL_DEVICE_MAX_COMPUTE_UNITS); }

    public long maxWorkItemSimensions() { return Query.getLong(_deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS); }

    public long[] maxWorkItemSizes() { return Query.getSizes(_deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3); }

    public long maxWorkGroupSize() { return Query.getSize(_deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE); }

    public long maxClockFrequenzy() { return Query.getLong(_deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY); }

    public int maxAddressBits() { return Query.getInt(_deviceId, CL_DEVICE_ADDRESS_BITS); }

    public long maxMemAllocSize() { return Query.getLong(_deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE); }

    public long globalMemSize() { return Query.getLong(_deviceId, CL_DEVICE_GLOBAL_MEM_SIZE); }

    public int errorCorrectionSupport() { return Query.getInt(_deviceId, CL_DEVICE_ERROR_CORRECTION_SUPPORT); }

    public int localMemType() { return Query.getInt(_deviceId, CL_DEVICE_LOCAL_MEM_TYPE); }

    public long localMemSize() { return Query.getLong(_deviceId, CL_DEVICE_LOCAL_MEM_SIZE); }

    public long maxConstantBufferSize() { return Query.getLong(_deviceId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE); }

    public long maxConstantBufferSizeKB() { return (int) (Query.getLong(_deviceId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) / 1024); }

    public int imageSupport() { return Query.getInt(_deviceId, CL_DEVICE_IMAGE_SUPPORT); }

    public int maxReadImageArgs() { return Query.getInt(_deviceId, CL_DEVICE_MAX_READ_IMAGE_ARGS); }

    public int maxWriteImageArgs() { return Query.getInt(_deviceId, CL_DEVICE_MAX_WRITE_IMAGE_ARGS); }

    public long singleFPConfig() { return Query.getLong(_deviceId, CL_DEVICE_SINGLE_FP_CONFIG); }

    public long image2DMaxWidth() { return Query.getSize(_deviceId, CL_DEVICE_IMAGE2D_MAX_WIDTH); }

    public long image2DMaxHeight() { return Query.getSize(_deviceId, CL_DEVICE_IMAGE2D_MAX_HEIGHT); }

    public long image3DMaxWidth() { return Query.getSize(_deviceId, CL_DEVICE_IMAGE3D_MAX_WIDTH); }

    public long image3DMaxHeight() { return Query.getSize(_deviceId, CL_DEVICE_IMAGE3D_MAX_HEIGHT); }

    public long image3DMaxDepth() { return Query.getSize(_deviceId, CL_DEVICE_IMAGE3D_MAX_DEPTH); }

    public int prefVecWidthChar() { return Query.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR); }

    public int prefVecWidthShort() { return Query.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT); }

    public int prefVecWidthInt() { return Query.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT); }

    public int prefVecWidthLong() { return Query.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG); }

    public int prefVecWidthFloat() { return Query.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT); }

    public int prefVecWidthDouble() { return Query.getInt(_deviceId, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE); }

    public static class Query {
        /**
         * Returns the value of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value
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
         * @return The value
         */
        public static int[] getInts(cl_device_id device, int paramName, int numValues) {
            int[] values = new int[numValues];
            clGetDeviceInfo(device, paramName, (long) Sizeof.cl_int * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         * Returns the value of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value
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
         * @return The value
         */
        public static long[] getLongs(cl_device_id device, int paramName, int numValues) {
            long[] values = new long[numValues];
            clGetDeviceInfo(device, paramName, (long) Sizeof.cl_long * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         * Returns the value of the device info parameter with the given name
         *
         * @param device    The device
         * @param paramName The parameter name
         * @return The value
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
         * Returns the value of the platform info parameter with the given name
         *
         * @param platform  The platform
         * @param paramName The parameter name
         * @return The value
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
         * Returns the value of the device info parameter with the given name
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
            ByteBuffer buffer = ByteBuffer.allocate(numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
            clGetDeviceInfo(
                    device,
                    paramName,
                    (long) Sizeof.size_t * numValues,
                    Pointer.to(buffer),
                    null
            );
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


    private <T extends Number> neureka.Data<T> _dataArrayOf( Object data, DataType<T> dataType ) {
        return (neureka.Data<T>) new CLData(this, data, (DataType<Number>) dataType);
    }

    private static class CLData extends DeviceData<Number> {

        public CLData(Device<Number> owner, Object dataRef, DataType<Number> dataType) {
            super(owner, dataRef, dataType);
            assert !(dataRef instanceof neureka.Data);
        }

        @Override
        public neureka.Data<Number> withNDConf(NDConfiguration ndc) {
            // We create a new cl_tsr object with the same data but a different ND-configuration:
            cl_tsr<?,?> clTsr = (cl_tsr<?,?>) _dataRef;
            cl_tsr.cl_config config = ((OpenCLDevice)_owner)._writeNDConfig( ndc );
            cl_tsr<?,?> newDataRef = new cl_tsr<>(clTsr.value, clTsr.dtype, config);
            return new CLData((Device<Number>) _owner, newDataRef, _dataType );
        }
    }

}
