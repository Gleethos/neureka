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


package neureka.backend.api.implementations;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.executions.ExecutorFor;
import neureka.devices.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.frontend.AbstractFunction;

import java.util.function.Consumer;
import java.util.function.Function;

/**
 *   This class is the middle layer of the 3 tier abstraction architecture
 *   of Neureka's operation implementations.
 *
 *   Conceptually an implementation of this interface represents "a way of execution" for
 *   the OperationType to which an instance of said implementation would belong.
 *   The "+" operator for example has different OperationTypeImplementation instances
 *   for different ExecutionCall instances.
 *   Tensors within an execution call having the same shape would
 *   trigger the Operation instance of the OperationType, whereas otherwise
 *   the Convolution or Broadcast implementation might be called.
 */
public interface OperationTypeImplementation<FinalType>
{
    String getName();

    interface SuitabilityChecker {
        float canHandle( ExecutionCall call );
    }

    float isImplementationSuitableFor( ExecutionCall call );

    interface DeviceFinder {
        Device findFor( ExecutionCall call );
    }

    Device findDeviceFor( ExecutionCall call );

    interface ForwardADAnalyzer {
        boolean allowsForward( ExecutionCall call );
    }

    boolean canImplementationPerformForwardADFor(ExecutionCall call );

    interface BackwardADAnalyzer {
        boolean allowsBackward( ExecutionCall call );
    }

    boolean canImplementationPerformBackwardADFor(ExecutionCall call );

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
        Tsr handle( AbstractFunction caller,  ExecutionCall call );
    }

    Tsr handleInsteadOfDevice(  AbstractFunction caller, ExecutionCall call );

    interface RecursiveJunctionAgent {
        Tsr handle( ExecutionCall call, Function<ExecutionCall, Tsr> goDeeperWith );
    }

    Tsr handleRecursivelyAccordingToArity( ExecutionCall call, Function<ExecutionCall, Tsr> goDeeperWith );

    interface DrainInstantiation {
        ExecutionCall handle( ExecutionCall call );
    }

    ExecutionCall instantiateNewTensorsForExecutionIn( ExecutionCall call );


    Tsr recursiveReductionOf(ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );

    <D extends Device, E extends ExecutorFor<D>> FinalType setExecutor(Class<E> deviceClass, E execution);

    <D extends Device, E extends ExecutorFor<D>> ExecutorFor getExecutor(Class<E> deviceClass);



}
