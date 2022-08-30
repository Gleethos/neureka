package neureka;

import neureka.devices.Device;

/**
 *  A wrapper type for the raw data array of a tensor/nd-array,
 *  which is provided by implementations of the {@link Device} interface.
 *  This is used to interface with the raw data as well as check where it comes from.
 *
 * @param <V> The type of the data array.
 */
public interface Data<V> {

    /**
     * @return The owner of this data array wrapper (the device which allocated the memory).
     */
    Device<V> owner();

    /**
     *  This returns the underlying raw data object of an nd-array or tensor.
     *  Contrary to the {@link Nda#getItems()} ()} method, this will
     *  return an unbiased view on the raw data of this tensor.
     *  Be careful using this, as it exposes mutable state!
     *
     * @return The raw data object underlying an nd-array/tensor.
     */
    Object getRef();

    default <D> D getRef(Class<D> dataType) {
        Object data = getRef();
        if ( data != null && !dataType.isAssignableFrom(data.getClass()) )
            throw new IllegalArgumentException("Provided data type '"+dataType+"' is not assignable from '"+data.getClass()+"'.");
        return (D) data;
    }

}
