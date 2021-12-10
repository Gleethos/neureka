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

import neureka.backend.api.BackendExtension;
import neureka.calculus.assembly.ParseUtil;
import neureka.common.composition.Component;
import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.BackendContext;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.host.CPU;
import neureka.ndim.AbstractNDArray;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Implementations of this represent computational
 * devices for storing tensors (instances of the Tsr&lt;V&gt; class), which may
 * also expose a useful API for executing operations on tensors.
 * Such instances are also components of tensors, which is why
 * this interface extends the Component&lt;Tsr&lt;V&gt;&gt; interface.
 *
 * The device interface extends the "{@link Storage}" interface because devices
 * are capable of storing tensors on them.
 * A tensor stored on a device holds a reference to that device,
 * as well as the device itself which may also know about the tensors it holds.
 * A tensor stored on a device will have its "isOutsourced" property set to true!
 *
 * @param <V> The value type of super type of the values stored on a {@link Device} implementation...
 */
public interface Device<V> extends Component<Tsr<V>>, Storage<V>, Iterable<Tsr<V>>
{
    /**
     * This method returns {@link Device} instances matching
     * the given search parameter.
     * @param searchKeys The search parameter and name of the requested {@link Device} instance.
     * @return The found {@link Device} instance or simply the {@link CPU} instance by default.
     */
    static Device<?> find( String... searchKeys )
    {
        return find( Device.class, searchKeys );
    }

    /**
     * This method returns {@link Device} instances matching
     * the given search parameter.
     * @param searchKeys The search parameter and name of the requested {@link Device} instance.
     * @return The found {@link Device} instance or simply the {@link CPU} instance by default.
     */
    static <T, D extends Device<T>> D find( Class<D> deviceType, String... searchKeys )
    {
        if ( deviceType == CPU.class ) return (D) CPU.get();
        String key;
        if ( searchKeys.length == 0 ) key = "";
        else key = String.join(" ", searchKeys).toLowerCase();

        boolean justTakeFirstOne = key.equals("first");

        boolean probablyWantsGPU = Stream.of(
                                        "gpu", "nvidia", "amd", "intel", "opencl", "fpga", "radeon", "cuda", "apu", "graphics"
                                    )
                                    .anyMatch(key::contains);

        double desireForCPU = Stream.of("jvm","native","host","cpu","threaded")
                                    .mapToDouble( word -> ParseUtil.similarity( word, key ) )
                                    .max()
                                    .orElse(0);

        if ( probablyWantsGPU ) desireForCPU /= 10; // CPU instance is most likely not meant!

        for ( BackendExtension extension : Neureka.get().backend().getExtensions() ) {
            BackendExtension.DeviceOption found = extension.find( key );
            if ( found != null && (deviceType.isAssignableFrom( found.device().getClass() )) ) {
                if ( found.confidence() > desireForCPU || justTakeFirstOne )
                    return (D) found.device();
            }
        }

        if ( probablyWantsGPU )
            return null; // User wants OpenCL but cannot have it :/
        else if ( deviceType.isAssignableFrom( CPU.class ) )
            return (D) CPU.get();
        else
            return null; // We don't know what the user wants, but we do not have it :/
    }

    /**
     *  This method signals the device to get ready for garbage collection.
     *  A given device may have resources which ought to be freed when it is no longer used.
     *  One may also choose to do resource freeing manually.
     */
    void dispose();

    /**
     *  Implementations of this method ought to store the value
     *  of the given tensor and the "parent" tensor in whatever
     *  formant suites the underlying implementation and or final type.
     *  {@link Device} implementations are also tensor storages
     *  which may also have to store tensors which are slices of bigger tensors.
     *
     * @param tensor The tensor whose data ought to be stored.
     * @return A reference this object to allow for method chaining. (factory pattern)
     */
    <T extends V> Device<V> store(Tsr<T> tensor, Tsr<T> parent );

    <T extends V> boolean has(Tsr<T> tensor );

    <T extends V> Device<V> free( Tsr<T> tensor );

    <T extends V> Device<V> cleaning( Tsr<T> tensor, Runnable action );

    <T extends V> Device<V> write( Tsr<T> tensor, Object value );

    <T extends V> Device<V> swap( Tsr<T> former, Tsr<T> replacement );

    Device<V> approve( ExecutionCall<? extends Device<?>> call );

    <T extends V> Device<V> updateNDConf( AbstractNDArray<?, T> tensor );

    <T extends V> Object valueFor( Tsr<T> tensor );

    <T extends V> V valueFor( Tsr<T> tensor, int index );

    Collection<Tsr<V>> getTensors();

    Operation optimizedOperationOf( Function function, String name );

    default Function optimizedFunctionOf( Function function, String name ) {
        Operation optimizedOperation = optimizedOperationOf( function, name );
        BackendContext currentContext = Neureka.get().backend();
        if ( !currentContext.hasOperation( optimizedOperation ) )
            currentContext.addOperation( optimizedOperation );

        return new FunctionBuilder( currentContext )
                            .build(
                                    optimizedOperation,
                                    function.numberOfArgs(),
                                    function.isDoingAD()
                            );
    }

    /**
     *  This is a very simple fluent API for temporarily storing a number
     *  of tensors on this {@link Device}, executing a provided lambda action,
     *  and then migrating all the tensors back to their original devices.
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
    default In use( Tsr<V> first, Tsr<V>... rest ) {
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


    interface In {
        /**
         *
         * @param lambda
         * @param <R> The return type parameter of the lambda which is expected to be passed to
         *            the context runner {@link In} returned by this method.
         *
         * @return
         */
        <R> R in( Supplier<R> lambda );
    }

}
