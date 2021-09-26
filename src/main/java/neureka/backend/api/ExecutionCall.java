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

   ______                     _   _              _____      _ _
  |  ____|                   | | (_)            / ____|    | | |
  | |__  __  _____  ___ _   _| |_ _  ___  _ __ | |     __ _| | |
  |  __| \ \/ / _ \/ __| | | | __| |/ _ \| '_ \| |    / _` | | |
  | |____ >  <  __/ (__| |_| | |_| | (_) | | | | |___| (_| | | |
  |______/_/\_\___|\___|\__,_|\__|_|\___/|_| |_|\_____\__,_|_|_|

    A simple class which wraps essential arguments and context data
    used for operation execution on Device instances.


*/

package neureka.backend.api;


import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.devices.Device;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *  This class is a simple container holding relevant
 *  arguments needed to execute on a targeted {@link Device} which
 *  is specified by the type parameter below. <br>
 *  <br>
 *  It also holds a context components responsible for storing operation specific variables.
 *  This is Certain operations might require additionally parameters then the ones
 *  defined in this class... <br>
 *
 * @param <D> The Device implementation targeted by an instance of this ExecutionCall!
*/
public class ExecutionCall<D extends Device<?>> extends Call<D>
{
    /**
     *  This is the operation type which will be applied to this execution call.
     *  It contains multiple implementations, one of which might be applicable to this call...
     */
    private final Operation _operation;

    /**
     *  This Algorithm variable is the chosen algorithm for a given execution call instance.
     *  The variable is initially null and will be chosen dynamically based on an access request
     *  to the corresponding getter method for this variable.
     *  So it is in essence a lazy load variable.
     *  Choosing an algorithm occurs through the {@link ExecutionCall#_operation} variable,
     *  which is of type {@link Operation} and contains multiple algorithms for different execution call scenarios...
     */
    private Algorithm<?> _algorithm = null;

    private ExecutionCall(
            D device,
            Operation operation,
            Tsr<?>[] tensors,
            Algorithm<?> algorithm,
            List<Arg> arguments
    ) {
        super(tensors, device, arguments);
        this._operation = operation;
        this._tensors = tensors;
        this._algorithm = algorithm;
    }

    public static <D extends Device<?>> Builder<D> of( Tsr<?>... tensors ) {
        return new Builder<D>(tensors);
    }

    /**
     *  Warning: This is the only method on this class which exposes
     *  mutability to parts of the internals of an {@link ExecutionCall}.
     *  Do not use it too extensively in order to keep complexity
     *  to a minimum...
     *
     * @param arg The meta argument which ought to be stored on this {@link ExecutionCall}.
     * @return This very instance to allow for method chaining.
     */
    public ExecutionCall<D> setMetaArg( Arg<?> arg ) { _arguments.set(arg); return this; }
    
    public String toString() {
        return "ExecutionCall(" +
                    "device=" + this._device + ", " +
                    "derivativeIndex=" + this.getValOf( Arg.DerivIdx.class ) + ", " +
                    "operation=" + this._operation + ", " +
                    "tensors=" + java.util.Arrays.deepToString(this._tensors) + ", " +
                    "j=" + this.getJ() + ", " +
                    "algorithm=" + this.getAlgorithm() + ", " +
                    "context=" + _arguments.getAll(Arg.class) +
                ")";
    }

    public Operation getOperation() { return this._operation; }

    public int getJ() { return this.getValOf( Arg.VarIdx.class ); }

    public ExecutionCall<D> withTensors(Tsr<?>[] _tensors ) {
        return this._tensors == _tensors
                ? this
                : new ExecutionCall<>(
                        this._device, this._operation,
                        _tensors, _algorithm, _arguments.getAll(Arg.class)
                    );
    }

    public interface TensorCondition    { boolean check( Tsr<?> tensor ); }
    public interface TensorCompare      { boolean check( Tsr<?> first, Tsr<?> second ); }
    public interface DeviceCondition    { boolean check( Device<?> device ); }
    public interface OperationCondition { boolean check( Operation type ); }

    public interface Mutator { Tsr<?>[] mutate( Tsr<?>[] tensors ); }

    public <T extends Device<?>> ExecutionCall<T> forDeviceType( Class<T> type ) {
        assert _device.getClass() == type;
        return (ExecutionCall<T>) this;
    }

    public Algorithm<?> getAlgorithm() {
        if ( _algorithm != null ) return _algorithm;
        else _algorithm = _operation.getAlgorithmFor( this );
        return _algorithm;
    }

    public boolean allowsForward() {
        return getAlgorithm().canPerformForwardADFor( this );
    }

    public boolean allowsBackward() {
        return getAlgorithm().canPerformBackwardADFor( this );
    }

    public ADAgent getADAgentFrom(
            Function function,
            ExecutionCall<? extends Device<?>> call,
            boolean forward
    ) {
        for ( Arg<?> arg : _arguments.getAll(Arg.class) ) {
            if ( !call._arguments.has(arg.getClass()) )
                call._arguments.set(arg);
            // else: This should not happen.
        }
        return getAlgorithm().supplyADAgentFor( function, call, forward );
    }

    public void mutateArguments( Mutator mutation ) {
        _tensors = mutation.mutate( _tensors );
    }

    public static class Builder<D extends Device<?>>
    {
        private Operation _operation;
        private Tsr<?>[] _tensors;
        private Algorithm<?> _algorithm;
        private final List<Arg> _arguments = Stream.of(
                                                Arg.DerivIdx.of(-1),
                                                Arg.VarIdx.of(-1)
                                            )
                                            .collect(Collectors.toList());

        private Builder(Tsr<?>[] tensors) { _tensors = tensors; }

        public <V, D extends Device<V>> ExecutionCall<D> on(D device) {
            return new ExecutionCall<>(
                                    device,
                                    _operation,
                                    _tensors,
                                    _algorithm,
                                    _arguments
                            );
        }

        public Builder<D> running(Operation operation) {
            _operation = operation;
            return this;
        }

        public Builder<D> algorithm(Algorithm<?> algorithm) {
            _algorithm = algorithm;
            return this;
        }

        public Builder<D> andArgs( List<Arg> context ) {
            _arguments.addAll(context);
            return this;
        }

        public Builder<D> andArgs( Arg<?>... context ) {
            return andArgs(Arrays.stream(context).collect(Collectors.toList()));
        }

    }

}