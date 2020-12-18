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


   ______                     _   _             ______
  |  ____|                   | | (_)           |  ____|
  | |__  __  _____  ___ _   _| |_ _  ___  _ __ | |__ ___  _ __
  |  __| \ \/ / _ \/ __| | | | __| |/ _ \| '_ \|  __/ _ \| '__|
  | |____ >  <  __/ (__| |_| | |_| | (_) | | | | | | (_) | |
  |______/_/\_\___|\___|\__,_|\__|_|\___/|_| |_|_|  \___/|_|


------------------------------------------------------------------------------------------------------------------------

*/


package neureka.backend.api.executions;

import neureka.devices.Device;
import neureka.backend.api.ExecutionCall;

/**
 * This interface describes the functionality of an implementation
 * of an execution procedure for a specific Device (interface) instance
 * and OperationTypeImplementation (interface) instance!
 *
 * An instance of this interface is then a component of an OperationTypeImplementation instance
 * which is itself a component of the OperationType class.
 *
 * @param <TargetDevice> The Device type for which an implementation of this interface has been made.
 */
public interface ExecutorFor< TargetDevice extends Device >
{
    /**
     * Every ExecutorFor &lt; Device &gt; implementation needs to also
     * implement a lambda defined by the interface below.
     * The lambda shall take the call arguments and call
     * the specific methods of the Device type implementation
     * in order to satisfy the OperationTypeImplementation to which
     * this class belongs.
     *
     * @param <TargetDevice>
     */
    interface ExecutionOn< TargetDevice extends Device >
    {
        void run(ExecutionCall<TargetDevice> call );
    }

    ExecutionOn< TargetDevice > getExecution();

    int arity();

}
