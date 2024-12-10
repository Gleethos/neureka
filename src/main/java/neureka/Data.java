package neureka;

import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;

/**
 *  A wrapper type for the raw data array of a tensor/nd-array,
 *  which is typically provided by implementations of the {@link Device} interface.
 *  Every tensor/nd-array has a {@link Data} object which it uses to access its raw data.
 *  Use this to access the raw data of a nd-array and to check where it currently resides.
 *  But be careful as this exposes mutable state as well as backend specific implementations
 *  and types (e.g. OpenCL / JVM arrays).
 *
 * @param <V> The type of the data array.
 */
public interface Data<V>
{
    /**
     *  This is a static factory method which returns a {@link Data} object
     *  which does not contain any data. It is a sort of no-operation null object
     *  which can be used to represent the absence of data.
     *  A deleted tensor will typically have a {@link Data} object which does not contain any data.
     *
     * @return A {@link Data} object which does not contain any data.
     */
    static Data<Void> none() { return NoOpData.INSTANCE; }


    static <V> Data<V> of( Class<V> type, V... data ) { return CPU.get().allocate( type, data ); }

    static Data<Float> of( float... items ) { return CPU.get().allocate( Float.class, items ); }

    static Data<Double> of( double... items ) { return CPU.get().allocate( Double.class, items ); }

    static Data<Integer> of( int... items ) { return CPU.get().allocate( Integer.class, items ); }

    static Data<Long> of( long... items ) { return CPU.get().allocate( Long.class, items ); }

    static Data<Byte> of( byte... items ) { return CPU.get().allocate( Byte.class, items ); }

    static Data<Short> of( short... items ) { return CPU.get().allocate( Short.class, items ); }

    static Data<Boolean> of( boolean... items ) { return CPU.get().allocate( Boolean.class, items ); }

    static Data<Character> of( char... items ) { return CPU.get().allocate( Character.class, items ); }

    static Data<String> of( String... items ) { return CPU.get().allocate( String.class, items ); }

    /**
     * @return The owner of this data array wrapper (the device which allocated the memory).
     */
    Device<V> owner();

    /**
     *  This returns the underlying raw data object of a nd-array or tensor
     *  of a backend specific type (e.g. OpenCL memory object or JVM array).
     *  Contrary to the {@link Nda#getItems()} ()} method, this will
     *  return an unbiased view on the raw data of this tensor.
     *  Be careful using this, as it exposes mutable state!
     *
     * @return The raw data object underlying a nd-array/tensor, or null if the data is not present.
     */
    Object getOrNull();

    /**
     *  This returns the underlying raw data object of a nd-array or tensor
     *  of a backend specific type (e.g. OpenCL memory object or JVM array).
     *  Contrary to the {@link Nda#getItems()} ()} method, this will
     *  return an unbiased view on the raw data of this tensor.
     *  Be careful using this, as it exposes mutable state!
     * @throws NullPointerException if the data reference is null.
     *
     * @return The raw data object underlying a nd-array/tensor.
     */
    default Object get() {
        Object data = getOrNull();
        if ( data == null ) throw new NullPointerException("The data reference is missing!");
        return data;
    }

    /**
     *  This returns the underlying raw data object of a nd-array or tensor.
     *  Contrary to the {@link Nda#getItems()} ()} method, this will
     *  return an unbiased view on the raw data of this tensor.
     *  Be careful using this, as it exposes mutable state!
     *
     * @param dataType The type the underlying reference object is expected to have (this may be a JVM array or something device specific).
     * @return The raw data object underlying a nd-array/tensor.
     */
    default <D> D as( Class<D> dataType ) {
        Object data = getOrNull();
        if ( data != null && !dataType.isAssignableFrom(data.getClass()) )
            throw new IllegalArgumentException("Provided data type '"+dataType+"' is not assignable from '"+data.getClass()+"'.");
        return dataType.cast(data);
    }

    /**
     * @return The data type of the raw data array.
     */
    DataType<V> dataType();

    /**
     *  This method returns the number of times this data object is currently in use by a nd-array,
     *  meaning that the number of usages is also the number of nd-arrays which are currently
     *  referencing this data object. <br>
     *  The reason why this can be greater than one is because of the existence of sliced, transposed
     *  and reshaped nd-arrays which all share the same data object as their parent nd-array.
     *
     * @return The number of times this data object is currently in use by a nd-array.
     */
    int usages();
}
