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

   _____             _
  |  __ \           (_)
  | |  | | _____   ___  ___ ___
  | |  | |/ _ \ \ / / |/ __/ _ \
  | |__| |  __/\ V /| | (_|  __/
  |_____/ \___| \_/ |_|\___\___|

    An abstract of a backend implementations which handles tensors, their data
    and executions on these tensors / their data.

*/

package neureka.devices;

import neureka.MutateTsr;
import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.BackendContext;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionParser;
import neureka.common.composition.Component;
import neureka.common.utility.LogUtil;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.ndim.config.NDConfiguration;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Implementations of this represent computational
 * devices for storing tensors (instances of the Tsr&lt;V&gt; class), which may
 * also expose a useful API for executing operations on tensors (used in backend operations).
 * Such instances are also components of tensors, which is why
 * this interface extends the Component&lt;Tsr&lt;V&gt;&gt; interface.                        <br><br>
 *
 * Because devices store tensors, this interface extends the "{@link Storage}" interface
 * which defines the API for storing them.
 * A tensor stored on a device holds a reference to that device,
 * as well as the device itself which may also know about the tensors it holds.
 * A tensor stored on a device will have its "isOutsourced" property set to true!
 *
 * @param <V> The super type of all values that can be stored on a {@link Device} implementation...
 */
public interface Device<V> extends Component<Tsr<V>>, Storage<V>, Iterable<Tsr<V>>
{
    /**
     * This method returns {@link Device} instances matching
     * the given search parameter.
     * @param searchKeys The search parameter and name of the requested {@link Device} instance.
     * @return The found {@link Device} instance or simply the {@link CPU} instance by default.
     */
    static Optional<Device<Object>> find( String... searchKeys ) {
        return Optional.ofNullable( Query.query( Device.class, searchKeys ) );
    }

    /**
     *  This method returns {@link Device} instances matching
     *  the given search parameters.
     *
     * @param deviceType The device type class which should be found.
     * @param searchKeys The search parameter and name of the requested {@link Device} instance.
     * @param <T> The value super types of the tensors stored on the requested device.
     * @param <D> The device type parameter.
     * @return The found {@link Device} instance or simply the {@link CPU} instance by default.
     */
    static <T, D extends Device<T>> Optional<D> find( Class<D> deviceType, String... searchKeys ) {
        return Optional.ofNullable( Query.query( deviceType, searchKeys ) );
    }

    /**
     *  This method returns {@link Device} instances matching
     *  the given search parameter.
     *  If the provided keys do not match anything then this method will simply return a {@link CPU} instance.
     *
     * @param searchKeys The search parameter and name of the requested {@link Device} instance.
     * @return The found {@link Device} instance or simply the {@link CPU} instance by default.
     */
    static Device<Object> any( String... searchKeys ) {
        Device<Object> found = get( Device.class, searchKeys );
        return ( found == null ? CPU.get() : found );
    }

    /**
     *  This method returns {@link Device} instances matching
     *  the given search parameter.
     *  If the provided keys do not match anything then this method may return null.
     *
     * @param searchKeys The search parameter and name of the requested {@link Device} instance.
     * @return The found {@link Device} instance or simply {@code null} by default.
     */
    static Device<Object> get( String... searchKeys ) {
        LogUtil.nullArgCheck( searchKeys, "searchKeys", String.class );
        return get( Device.class, searchKeys );
    }

    /**
     *  This method returns {@link Device} instances matching
     *  the given search parameters.
     *
     * @param deviceType The device type class which should be found.
     * @param searchKeys The search parameter and name of the requested {@link Device} instance.
     * @param <T> The value super types of the tensors stored on the requested device.
     * @param <D> The device type parameter.
     * @return The found {@link Device} instance or null if nothing was found which matches the provided search hints well enough.
     */
    static <T, D extends Device<T>> D get( Class<D> deviceType, String... searchKeys ) {
        LogUtil.nullArgCheck( deviceType, "deviceType", Class.class );
        LogUtil.nullArgCheck( searchKeys, "searchKeys", String.class );
        if ( searchKeys.length == 0 ) searchKeys = new String[] { "first" };
        return Query.query( deviceType, searchKeys );
    }

    /**
     *  This method signals the device to get ready for garbage collection.
     *  A given device may have resources which ought to be freed when it is no longer used.
     *  One may also choose to do resource freeing manually.
     */
    void dispose();

    /**
     *  Use this to check if a tensor is stored on this {@link Device}!  <br><br>
     *
     * @param tensor The tensor which may or may not be stored on this {@link Device}.
     * @param <T> The type parameter for the value type of the tensor, which must be supported by this {@link Device}.
     * @return The truth value determining if the provided tensor is stored on this {@link Device}.
     */
    <T extends V> boolean has( Tsr<T> tensor );

    /**
     *  Use this to remove the provided tensor from this {@link Device}!  <br><br>
     *
     * @param tensor The tensor which ought to be removed from this {@link Device}.
     * @param <T> The type parameter for the value type of the tensor, which must be supported by this {@link Device}.
     * @return This very instance to allow for method chaining.
     */
    <T extends V> Device<V> free( Tsr<T> tensor );

    /**
     *  This method exposes the tensor access API for reading from or writing to
     *  a tensor stored on this device.
     *  It may return null if this device does not support
     *  accessing stored tensors.
     *
     * @param tensor The tensor whose data ought to be accessed.
     * @param <T> The type parameter of the tensor for which the access API should be returned.
     * @return The tensor access API for reading from or writing to a tensor stored on this device.
     */
    <T extends V> Access<T> access( Tsr<T> tensor );

    /**
     *  This method is used internally to give {@link Device} implementations the opportunity
     *  to perform some exception handling before the {@link ExecutionCall} will be dispatched.
     *  Use this for debugging when doing custom backend operations.
     *
     * @param call The {@link ExecutionCall} which should be approved by this {@link Device} before execution.
     * @return This very instance to allow for method chaining.
     */
    Device<V> approve( ExecutionCall<? extends Device<?>> call );

    /**
     * @return A {@link Collection} of all tensors stored by this device.
     */
    Collection<Tsr<V>> getTensors();

    <T extends V> neureka.Data<T> allocate(DataType<T> dataType, int size );

    <T extends V> neureka.Data<T> allocate(DataType<T> dataType, int size, T initialValue );

    neureka.Data<Object> allocate( Object jvmData, int desiredSize );

    /**
     *  This method tries to allow this device to produce an optimized {@link Operation}
     *  based on the provided function.
     *  This is especially useful in an OpenCL context which can compile the function
     *  into native GPU kernels at runtime.
     *
     * @param function The function which should be turned into an optimized operation.
     * @param name The name of the returned operation.
     * @return An optimized operation based on the provided function, or null if optimization is not possible.
     */
    Operation optimizedOperationOf( Function function, String name );

    /**
     *  This method tries to allow this device to produce an optimized {@link Function}
     *  based on the provided function.
     *  This is especially useful in an OpenCL context which can compile the function
     *  into native GPU kernels at runtime.
     *
     * @param function The function which should be used to design a new optimized function.
     * @param name The name of the optimized operation underlying the returned function.
     * @return An instance of the optimized function.
     */
    default Function optimizedFunctionOf( Function function, String name ) {
        LogUtil.nullArgCheck( function, "function", Function.class );
        LogUtil.nullArgCheck( name, "name", String.class );

        Operation optimizedOperation = optimizedOperationOf( function, name );
        BackendContext currentContext = Neureka.get().backend();
        if ( !currentContext.hasOperation( optimizedOperation ) )
            currentContext.addOperation( optimizedOperation );

        return new FunctionParser( currentContext )
                            .parse(
                                optimizedOperation,
                                function.numberOfArgs(),
                                function.isDoingAD()
                            );
    }

    /**
     *  This is a very simple fluent API for temporarily storing a number
     *  of tensors on this {@link Device}, executing a provided lambda action,
     *  and then migrating all the tensors back to their original devices.              <br><br>
     *
     * @param first The first tensor among all passed tensors which ought to be
     *              stored temporarily on this {@link Device}.
     * @param rest Any number of other tensors passed to this method to be
     *             stored temporarily on this {@link Device}.
     *
     * @return A simple lambda runner which will migrate the tensors passed to this method to
     *         this very {@link Device}, execute the provided lambda, and then  migrate all the
     *         tensors back to their original devices!
     */
    default In borrow( Tsr<V> first, Tsr<V>... rest ) {
        LogUtil.nullArgCheck( first, "first", Tsr.class );
        LogUtil.nullArgCheck( rest, "rest", Tsr[].class );
        List<Tsr<V>> tensors = new ArrayList<>();
        if ( first != null ) tensors.add( first );
        if ( rest.length > 0 )
            tensors.addAll( Arrays.stream( rest ).filter(Objects::nonNull).collect(Collectors.toList()) );
        Device<?> thisDevice = this;
        return new In() {
            @Override
            public <R> R in( Supplier<R> lambda ) {
                List<Device<?>> devices = tensors.stream().map( Tsr::getDevice ).collect( Collectors.toList() );
                for ( Tsr<V> t : tensors ) t.to( thisDevice );
                R result = lambda.get();
                for ( int i = 0; i < tensors.size(); i++ ) {
                    if ( devices.get( i ) != null ) tensors.get( i ).to( devices.get( i ) );
                }
                return result;
            }
        };
    }

    /**
     *  The second part of the method chain of the fluent API for executing
     *  tensors on this {@link Device} temporarily.
     */
    interface In
    {
        /**
         * @param lambda The lambda during which the previously provided tensors should be stored on this {@link Device}.
         * @param <R> The return type parameter of the lambda which is expected to be passed to
         *            the context runner {@link In} returned by this method.
         *
         * @return The return value, which may be anything.
         */
        <R> R in( Supplier<R> lambda );
    }

    interface Data<V>
    {
        /**
         *  Use this to write a single scalar item into the accessed tensor at
         *  one or more positions within the tensor.
         *
         * @param item The item which should be written to the tensor.
         * @return A {@link Device.Writer} implementation which expects the type of writing to be specified.
         */
        Writer write( V item );
        /**
         *  Use this to write data from an array into the accessed tensor.
         *
         * @param array The data array whose data should be britten from.
         * @param offset The start index offset within the provided data array.
         * @return A {@link Device.Writer} implementation which expects the type of writing to be specified.
         */
        Writer writeFrom( Object array, int offset );
        /**
         *  Use this method to write data to the provided tensor, given that
         *  the tensor is already stored on this device!                         <br><br>
         *
         * @param array The data inn the form of a primitive array.
         */
        default void writeFrom( Object array ) { this.writeFrom( array, 0 ).fully(); }
        /**
         *  Find a particular tensor item by providing its location.
         *
         * @param index The index at which a tensor item should be read and returned.
         * @return The tensor item found at the provided location.
         */
        V readAt( int index );
        /**
         *  Use this to read an array of items from the accessed tensor
         *  by specifying a start position of the chunk of data that should be read.
         *
         * @param arrayType The type of (primitive) array which should be read.
         * @param start The start position of the read cursor.
         * @param size The number of items which should be read from the tensor.
         * @param <A> The array type parameter specified by the provided class.
         * @return An instance of the provided array type class.
         */
        <A> A readArray( Class<A> arrayType, int start, int size );
        /**
         *  Use this to read the full data array of the accessed tensor.
         *
         * @param clone The truth value determining if the tensor should be copied or not.
         * @return The full data array of the tensor accessed by this API.
         */
        Object readAll( boolean clone );
        /**
         * @return The size of the underlying data array of the accessed tensor.
         */
        int getDataSize();

    }


    /**
     *  Implementations of this represent the access to tensors stored on a device
     *  in order to read from or write to said tensor. <br>
     *  <b>Warning: This API exposes the true underlying data of a tensor
     *  which does not take into account slice, reshape or stride information...</b>
     *
     * @param <V> The type parameter of the tensor accessed by an instance of this.
     */
    interface Access<V> extends Data<V>
    {
        /**
         *  Use this to perform some custom memory cleanup for when the accessed {@link Tsr} gets garbage collected.   <br><br>
         *
         * @param action The {@link Runnable} action which ought to be performed when the tensor gets garbage collected.
         */
        void cleanup( Runnable action );
        /**
         *  This method automatically called within the {@link MutateTsr#setNDConf(NDConfiguration)} method
         *  so that an outsourced tensor has a consistent ND-Configuration both in RAM and on any
         *  given {@link Device} implementation... <br><br>
         */
        void updateNDConf();

        neureka.Data actualize();
    }

    /**
     *  Instances of this complete a request for writing to an accessed tensor stored on a device.
     *  One may write at a particular position in a tensor, a range of positions or write
     *  to every possible value.
     */
    interface Writer
    {
        /**
         *  Writes whatever kind of data was previously specified, to the tensors'
         *  data at the position targeted by the provided {@code index}.
         *
         * @param index The position at which data should be written to.
         */
        default void at( int index ) { intoRange( index, index + 1 ); }
        /**
         *  Writes whatever kind of data was previously specified, to the tensors'
         *  data into the range targeted by the provided {@code start} and {@code limit}.
         *
         * @param start The first position of the writing cursor in the accessed tensor.
         * @param limit The exclusive limit of the range which should be written to.
         */
        void intoRange( int start, int limit );
        /**
         *  A convenience method for specifying that the entire data array of
         *  the accessed tensor should be written to.
         *  This is equivalent to calling {@link #intoRange(int, int)} with the arguments
         *  {@code 0} and {@code tensor.size()}.
         */
        void fully();
    }


}
