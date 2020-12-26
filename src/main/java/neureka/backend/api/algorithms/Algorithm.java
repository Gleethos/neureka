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


   _____                    _   _          _______            _____               _                           _        _   _
  / __  \                  | | (_)        |__   __|          |_   _|             | |                         | |      | | (_)
 | |  |_|__   ___ _ __ __ _| |_ _  ___  _ __ | L   _ _ __  ___ | | _ __ ___  _ __| | ___ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __
 | |  | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \|| | | | '_ \/ _ \| || '_ ` _ \| '_ \ |/ _ \ '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \
 | |__| |_) |  __/ | | (_| | |_| | (_) | | | || |_| | |_) | __/| || | | | | | |_) ||  __/ | | | | |  __/ | | | || (_| | |_| | (_) | | | |
  \___| .__/ \___|_|  \__,_|\__|_|\___/|_| |_||\__, | .__/\___|___|_| |_| |_| .__/_|\___|_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
      | |                                       __/ | |                     | |
      |_|                                      |___/|_|                     |_|

------------------------------------------------------------------------------------------------------------------------
*/


package neureka.backend.api.algorithms;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.implementations.ImplementationFor;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;
import neureka.autograd.ADAgent;

import java.util.function.Consumer;
import java.util.function.Function;

/**
 *   This class is the middle layer of the 3 tier abstraction architecture
 *   of Neureka's operation backend.
 *
 *   Conceptually an implementation of this interface represents "a way of execution" for
 *   the Operation to which an instance of said implementation would belong.
 *   The "+" operator for example has different Algorithm instances
 *   for different ExecutionCall instances.
 *   Tensors within an execution call having the same shape would
 *   trigger the Operation instance of an elementwise Algorithm instance, whereas otherwise
 *   the Convolution or Broadcast implementation might be called.
 */
public interface Algorithm<FinalType>
{
    String getName();

    interface SuitabilityChecker {
        float canHandle( ExecutionCall call );
    }

    float isAlgorithmSuitableFor(ExecutionCall call );

    interface DeviceFinder {
        Device findFor( ExecutionCall call );
    }

    Device findDeviceFor( ExecutionCall call );

    interface ForwardADAnalyzer {
        boolean allowsForward( ExecutionCall call );
    }

    boolean canAlgorithmPerformForwardADFor( ExecutionCall call );

    interface BackwardADAnalyzer {
        boolean allowsBackward( ExecutionCall call );
    }

    boolean canAlgorithmPerformBackwardADFor( ExecutionCall call );

    interface ADAgentSupplier {
        ADAgent getADAgentOf(
                neureka.calculus.Function f,
                ExecutionCall<Device> call,
                boolean forward
        );
    }

    ADAgent supplyADAgentFor(
            neureka.calculus.Function f,
            ExecutionCall<Device> call,
            boolean forward
    );

    interface InitialCallHook {
        Tsr handle( FunctionNode caller,  ExecutionCall call );
    }

    Tsr handleInsteadOfDevice( FunctionNode caller, ExecutionCall call );

    interface RecursiveJunctionAgent {
        Tsr handle( ExecutionCall call, Function<ExecutionCall, Tsr<?>> goDeeperWith );
    }

    Tsr handleRecursivelyAccordingToArity( ExecutionCall call, Function<ExecutionCall, Tsr<?>> goDeeperWith );

    interface DrainInstantiation {
        ExecutionCall handle( ExecutionCall call );
    }

    ExecutionCall instantiateNewTensorsForExecutionIn( ExecutionCall call );


    Tsr recursiveReductionOf( ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );

    /**
     *
     *  Implementations of the Algorithm interface ought to express a compositional design pattern. <br>
     *  This means that concrete implementations of an algorithm for a device are not extending
     *  an Algorithm, they are components of it instead. <br>
     *  These components can be stored on an Algorithm by passing
     *  a Device class as key and an ImplementationFor instance as value.
     *
     *
     * @param deviceClass
     * @param execution
     * @param <D>
     * @param <E>
     * @return
     */
    <D extends Device<?>, E extends ImplementationFor<D>> FinalType setImplementationFor(Class<D> deviceClass, E execution);

    /**
     *  A device specific implementation can be accessed by passing the class of the implementation
     *  of the 'ImplementationFor<Device>' class.
     *  An Algorithm instance ought to contain a collection of these device specific
     *  implementations...
     *
     * @param deviceClass The class of the device for which the stored algorithm implementation should be returned.
     * @param <D> The type parameter which has to be a class extending the Device interface.
     * @param <I> The type parameter for a class implementing the "ImplementationFor" interface.
     * @return The implementation for the passed device type class.
     */
    <D extends Device<?>> ImplementationFor<D> getImplementationFor(Class<D> deviceClass );


}
