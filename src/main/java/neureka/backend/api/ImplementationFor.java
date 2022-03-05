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


  _____                 _                           _        _   _             ______
 |_   _|               | |                         | |      | | (_)           |  ____|
   | |  _ __ ___  _ __ | | ___ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __ | |__ ___  _ __
   | | | '_ ` _ \| '_ \| |/ _ \ '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \|  __/ _ \| '__|
  _| |_| | | | | | |_) | |  __/ | | | | |  __/ | | | || (_| | |_| | (_) | | | | | | (_) | |
 |_____|_| |_| |_| .__/|_|\___|_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|_|  \___/|_|
                 | |
                 |_|

------------------------------------------------------------------------------------------------------------------------

*/


package neureka.backend.api;

import neureka.Tsr;
import neureka.devices.Device;

/**
 * Generally speaking, this interface describes the functionality of an implementation
 * of an execution procedure tailored to a specific {@link Device} (interface) instance
 * and {@link Algorithm} (interface) instance!
 * Instances of implementations of the {@link ImplementationFor} interface are components
 * of instances of implementations of the {@link Algorithm} interface,
 * which themselves are components of {@link Operation} implementation instances.
 *
 *
 * @param <TargetDevice> The Device type for which an implementation of this interface has been made.
 */
@FunctionalInterface
public interface ImplementationFor< TargetDevice extends Device<?> >
{
    /**
     *  This method is the entrypoint for a concrete implementation
     *  of the algorithm to which instances of this interface
     *  belong and the device on which this is implemented.
     *  One has to keep in mind that the implementation details
     *  with respect to the target device are specific to the
     *  methods of the "TargetDevice" type on which this call should run...
     *
     *  @param call The call which ought to be executed on this implementation.
     */
    void run( ExecutionCall<TargetDevice> call );

    default Tsr<?> runAndGetFirstTensor( ExecutionCall<TargetDevice> call ) {
        this.run( call );
        return call.input( 0 );
    }

}
