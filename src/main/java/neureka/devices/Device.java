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

import neureka.Component;
import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.OperationContext;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.CLContext;
import neureka.devices.opencl.OpenCLDevice;
import neureka.devices.opencl.OpenCLPlatform;

import java.util.Collection;
import java.util.function.IntFunction;
import java.util.stream.Stream;

/**
 * This is the interface for implementations representing
 * devices primarily store tensors, namely instances of the Tsr&lt;ValType&gt; class.
 * Optionally they might also be capable of executing operations on tensors.
 * Such instances are also components of tensors, which is why
 * this interface extends the Component &lt; Tsr &lt; ValType&gt; &gt; interface.
 *
 * The device interface extends the "Storage" interface because devices
 * are also capable of storing tensors on them.
 * A tensor stored on a device holds a reference to that device,
 * as well as the device itself which also knows about the tensors it holds.
 * A tensor stored on a device will have its "isOutsourced" property set to true!
 *
 */
public interface Device<ValType> extends Component<Tsr<ValType>>, Storage<ValType>, Iterable<Tsr<ValType>>
{
    /**
     * This method returns {@link Device} instances matching
     * the given search parameter.
     * @param name The search parameter and name of the requested {@link Device} instance.
     * @return The found {@link Device} instance or simply the HostCPU instance by default.
     */
    static Device<?> find( String name )
    {
        String search = name.toLowerCase();
        boolean probablyWantsGPU = Stream.of("gpu", "nvidia", "amd", "intel", "opencl", "fpga", "radeon")
                                            .anyMatch(search::contains);

        if ( !Neureka.get().canAccessOpenCL() ) {
            if ( probablyWantsGPU ) {
                return null; // User wants OpenCL but cannot have it :/
            } else return HostCPU.instance();
        }

        Device<Number> result = HostCPU.instance();
        double score = FunctionParser.similarity( "jvm native host cpu threaded", search );
        if ( probablyWantsGPU ) score /= 10; // HostCPU instance is most likely not meant!

        for ( OpenCLPlatform p : Neureka.get().context().find(CLContext.class).getPlatforms() ) {
            for ( OpenCLDevice d : p.getDevices() ) {
                String str = ("opencl | "+d.type()+" | "+d.name()+" | "+d.vendor()).toLowerCase();
                double similarity = FunctionParser.similarity( str, search );
                if ( similarity > score || str.contains(search) ) {
                    result = d;
                    score = similarity;
                }
            }
        }
        if ( result == HostCPU.instance() && name.equals("first") ) {
            Device<Number> first = Neureka.get()
                                            .context()
                                            .find(CLContext.class)
                                            .getPlatforms()
                                            .get( 0 )
                                            .getDevices()
                                            .get( 0 );

            if ( first!=null ) result = first;
        }
        return result;
    }

    /**
     *  This method signals the device to get ready for garbage collection.
     *  A given device may have resources which ought to be freed when it is no longer used.
     *  One may also chose to do resource freeing manually.
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
    <T extends ValType> Device<ValType> store( Tsr<T> tensor, Tsr<T> parent );

    <T extends ValType> boolean has( Tsr<T> tensor );

    <T extends ValType> Device<ValType> free( Tsr<T> tensor );

    Device<ValType> cleaning( Tsr<ValType> tensor, Runnable action );

    Device<ValType> overwrite64( Tsr<ValType> tensor, double[] value );

    Device<ValType> overwrite32( Tsr<ValType> tensor, float[] value );

    Device<ValType> swap( Tsr<ValType> former, Tsr<ValType> replacement );

    Device<ValType> execute( ExecutionCall<Device<?>> call );

    Object valueFor( Tsr<ValType> tensor );

    ValType valueFor( Tsr<ValType> tensor, int index );

    Collection< Tsr<ValType> > getTensors();

    /**
     *  This method has the same signature of the Collection interface in Java 11,
     *  however in order to enable Java 8 support as well
     *  the method below is a substitution.
     *
     * @param generator
     * @param <T>
     * @return
     */
    <T> T[] toArray( IntFunction<T[]> generator );

    Operation optimizedOperationOf( Function function, String name );

    default Function optimizedFunctionOf( Function function, String name ) {
        Operation optimizedOperation = optimizedOperationOf( function, name );
        OperationContext currentContext = Neureka.get().context();
        if ( !currentContext.hasOperation( optimizedOperation ) )
            currentContext.addOperation( optimizedOperation );

        return new FunctionBuilder( currentContext )
                            .build(
                                    optimizedOperation,
                                    function.numberOfArgs(),
                                    function.isDoingAD()
                            );
    }

}
