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
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    private final static Logger _LOG = LoggerFactory.getLogger(ExecutionCall.class);

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
        _operation = operation;
        _tensors   = tensors;
        _algorithm = algorithm;
        int thisArity = _tensors.length;
        if ( _operation != null && thisArity < Math.abs(_operation.getArity()) ) {
            throw new IllegalArgumentException(
                    "Trying to instantiate an '"+this.getClass().getSimpleName()+"' with an arity " +
                    "of "+thisArity+", which is not suitable for the targeted operation '" +
                    _operation.getClass().getSimpleName()+"' with" +
                    ( _operation.getArity() < 0 ? " a minimum " : " the expected " ) +
                    "arity of "+Math.abs(_operation.getArity()) + "."
            );
        }
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
        String algorithmString = "?";
        if ( _algorithm != null ) {
            algorithmString = _algorithm.toString();
        }
        return this.getClass().getSimpleName()+"[" +
                    "device="          + _device + "," +
                    "derivativeIndex=" + getValOf( Arg.DerivIdx.class ) + "," +
                    "operation="       + _operation + "," +
                    "tensors="         + "[.." + _tensors.length + "..]," +
                    "j="               + getJ() + ", " +
                    "algorithm="       + algorithmString + "," +
                    "context="         + _arguments.getAll(Arg.class) +
                "]";
    }

    public Operation getOperation() { return _operation; }

    public int getJ() { return this.getValOf( Arg.VarIdx.class ); }

    public ExecutionCall<D> withTensors( Tsr<?>... tensors ) {
        return _tensors == tensors
                ? this
                : new ExecutionCall<>(
                        _device, _operation,
                        tensors, _algorithm, _arguments.getAll(Arg.class)
                    );
    }

    public ExecutionCall<D> withArgs( Arg<?>... args ) {
        List<Arg> old = _arguments.getAll(Arg.class);
        old.addAll(Arrays.stream(args).collect(Collectors.toList()));
        return new ExecutionCall<>(
                    _device, _operation, _tensors, _algorithm, old
                );
    }

    public <T extends Device<?>> ExecutionCall<T> forDeviceType( Class<T> type ) {
        assert _device.getClass() == type;
        return (ExecutionCall<T>) this;
    }

    /**
     *  An {@link ExecutionCall} will either already have a targeted {@link Algorithm} defined
     *  at instantiation or otherwise it will query the associated {@link Operation}
     *  for an {@link Algorithm} best suitable for the state of this {@link ExecutionCall}.
     *  Generally speaking, this method should only very rarely return null, however, if it
     *  does, then this most definitely means that there is nor backend support
     *  for this call for execution...
     *
     * @return The {@link Algorithm} suitable for this {@link ExecutionCall}.
     */
    public Algorithm<?> getAlgorithm() {
        if ( _algorithm != null ) return _algorithm;
        else _algorithm = _operation.getAlgorithmFor( this );
        if ( _algorithm == null ) {
            _LOG.error(
                "No suitable '"+Algorithm.class.getSimpleName()+"' implementation found for this '"+this+"'!"
            );
        }
        return _algorithm;
    }

    /**
     *  This method queries the underlying {@link Operation} of this {@link ExecutionCall} to see
     *  if forward mode auto differentiation can be performed.
     *
     * @return The truth value determining if forward mode auto differentiation can be performed for this.
     */
    public boolean allowsForward() {
        Algorithm<?> algorithm = getAlgorithm();
        if ( algorithm != null )
            return algorithm.canPerformForwardADFor( this );
        return false;
    }

    /**
     *  This method queries the underlying {@link Operation} of this {@link ExecutionCall} to see
     *  if backward mode auto differentiation can be performed.
     *
     * @return The truth value determining if backward mode auto differentiation can be performed for this.
     */
    public boolean allowsBackward() {
        Algorithm<?> algorithm = getAlgorithm();
        if ( algorithm != null )
            return algorithm.canPerformBackwardADFor( this );
        return false;
    }

    public ADAgent getADAgentFrom(
            Function function,
            boolean forward
    ) {
        return getAlgorithm().supplyADAgentFor( function, this, forward );
    }

    /**
     * @param <D> The type parameter for the device targeted by the {@link ExecutionCall} built by this builder.
     */
    public static class Builder<D extends Device<?>>
    {
        private Operation _operation;
        private final Tsr<?>[] _tensors;
        private Algorithm<?> _algorithm;
        private final List<Arg> _arguments = Stream.of(
                                                Arg.DerivIdx.of(-1),
                                                Arg.VarIdx.of(-1)
                                            )
                                            .collect(Collectors.toList());

        private Builder(Tsr<?>[] tensors) { _tensors = tensors; }

        public <V, D extends Device<V>> ExecutionCall<D> on(D device) {
            return new ExecutionCall<>(
                                    device, _operation,
                                    _tensors, _algorithm, _arguments
                                );
        }

        public Builder<D> running( Operation operation ) {
            _operation = operation;
            return this;
        }

        public Builder<D> algorithm( Algorithm<?> algorithm ) {
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