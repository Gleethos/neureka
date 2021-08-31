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
 * @param <DeviceType> The Device implementation targeted by an instance of this ExecutionCall!
*/
public class ExecutionCall<DeviceType extends Device<?>>
{
    /**
     *  This field references the device on which this ExecutionCall should be executed.
     */
    private final DeviceType _device;

    /**
     *  This is the operation type which will be applied to this execution call.
     *  It contains multiple implementations, one of which might be applicable to this call...
     */
    private final Operation _operation;

    /**
     *  The tensor arguments from which an operation will either
     *  read or to which it will write. <br>
     *  The first entry of this array is usually containing the output tensor,
     *  however this is not a necessity.
     *  Some operation algorithms might use multiple argument entries as output tensors.
     */
    private Tsr<?>[] _tensors;

    /**
     *  This Algorithm variable is the chosen algorithm for a given execution call instance.
     *  The variable is initially null and will be chosen dynamically based on an access request
     *  to the corresponding getter method for this variable.
     *  So it is in essence a lazy load variable.
     *  Choosing an algorithm occurs through the {@link ExecutionCall#_operation} variable,
     *  which is of type {@link Operation} and contains multiple algorithms for different execution call scenarios...
     */
    private Algorithm<?> _algorithm = null;

    private final Args _arguments = new Args();

    private ExecutionCall(
            DeviceType device,
            Operation operation,
            Tsr<?>[] tensors,
            Algorithm<?> algorithm,
            List<Arg> context
    ) {
        this._device = device;
        this._operation = operation;
        this._tensors = tensors;
        this._algorithm = algorithm;
        for ( Arg<?> arg : context ) this.getMetaArgs().set(arg);
    }

    public static <DeviceType extends Device<?>> Builder<DeviceType> builder() { return new Builder<>(); }

    public static <D extends Device<?>> Builder<D> of( Tsr<?>... tensors ) {
        return new Builder<D>().tensors(tensors);
    }

    public Args getMetaArgs() { return _arguments; }
    
    public String toString() {
        return "ExecutionCall(" +
                    "device=" + this._device + ", " +
                    "derivativeIndex=" + this.getDerivativeIndex() + ", " +
                    "operation=" + this._operation + ", " +
                    "tensors=" + java.util.Arrays.deepToString(this._tensors) + ", " +
                    "j=" + this.getJ() + ", " +
                    "algorithm=" + this.getAlgorithm() + ", " +
                    "context=" + _arguments.getAll(Arg.class) +
                ")";
    }

    public DeviceType getDevice() {
        return this._device;
    }

    public <T> Device<T> getDeviceFor( Class<T> supportCheck ) {
        // TODO: Make it possible to query device for type support!
        return (Device<T>) this._device;
    }

    public int getDerivativeIndex() {
        return this.getMetaArgs().valOf( Arg.DerivIdx.class );
    }

    public Operation getOperation() { return this._operation; }

    public Tsr<?>[] getTensors() { return this._tensors; }

    public int getJ() { return this.getMetaArgs().valOf( Arg.VarIdx.class ); }

    public ExecutionCall<DeviceType> withTensors( Tsr<?>[] _tensors ) {
        return this._tensors == _tensors
                ? this
                : new ExecutionCall<>(
                        this._device, this._operation,
                        _tensors, _algorithm, _arguments.getAll(Arg.class)
                    );
    }

    public ExecutionCall<DeviceType> withJ( int j ) {
        List<Arg> args = _arguments.getAll(Arg.class);
        args.add(Arg.VarIdx.of(j));
        return this.getJ() == j
                ? this
                : new ExecutionCall<>( _device, _operation, _tensors, _algorithm, args );
    }


    public <V, T extends Arg<V>> V getValOf( Class<T> argumentClass ) {
        return _arguments.valOf(argumentClass);
    }

    public interface TensorCondition    { boolean check( Tsr<?> tensor ); }
    public interface TensorCompare      { boolean check( Tsr<?> first, Tsr<?> second ); }
    public interface DeviceCondition    { boolean check( Device<?> device ); }
    public interface OperationCondition { boolean check( Operation type ); }

    public interface Mutator { Tsr<?>[] mutate( Tsr<?>[] tensors ); }

    /**
     * Constructs a copy of this call with the provided device!
     */
    public ExecutionCall<? extends Device<?>> withDevice( Device<?> newDevice ) {
        return ExecutionCall.of( _tensors )
                                .andArgs( Arg.DerivIdx.of( getDerivativeIndex() ) )
                                .andArgs( _arguments.getAll(Arg.class) )
                                .running( _operation )
                                .algorithm( _algorithm )
                                .on( newDevice );
    }

    public <T extends Device<?>> ExecutionCall<T> forDeviceType( Class<T> type ) {
        assert _device.getClass() == type;
        return (ExecutionCall<T>) this;
    }


    public <V> Tsr<V> getTsrOfType( Class<V> valueTypeClass, int i ) {
        if ( valueTypeClass == null ) {
            throw new IllegalArgumentException(
                    "The provided tensor type class is null!\n" +
                    "Type safe access to the tensor parameter at index '"+i+"' failed."
            );
        }
        if ( _tensors[ i ] != null ) {
            Class<?> tensorTypeClass = _tensors[ i ].getValueClass();
            if ( !valueTypeClass.isAssignableFrom(tensorTypeClass) ) {
                throw new IllegalArgumentException(
                        "The item value type of the tensor stored at parameter position '"+i+"' is " +
                        "'"+tensorTypeClass.getSimpleName()+"' and is not a sub-type of the provided " +
                        "type '"+valueTypeClass.getSimpleName()+"'."
                );
            }
        }
        return (Tsr<V>) _tensors[ i ];
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

    public ADAgent getADAgentFrom( Function function, ExecutionCall<? extends Device<?>> call, boolean forward )
    {
        for ( Arg<?> arg : _arguments.getAll(Arg.class) ) {
            if ( !call.getMetaArgs().has(arg.getClass()) )
                call.getMetaArgs().set(arg);
            // else: This should not happen.
        }
        return getAlgorithm().supplyADAgentFor( function, call, forward );
    }

    public void mutateArguments( Mutator mutation ) {
        _tensors = mutation.mutate( _tensors );
    }

    public Validator validate() { return new Validator(); }

    public static class Builder<D extends Device<?>>
    {
        private D device;
        private Operation operation;
        private Tsr<?>[] tensors;
        private Algorithm<?> algorithm;
        private final List<Arg> context = Stream.of(
                                                Arg.DerivIdx.of(-1),
                                                Arg.VarIdx.of(-1)
                                            )
                                            .collect(Collectors.toList());

        Builder() { }


        // ExecutionCall.of(t1, t2).andArgs().running(operation).on(device);

        public Builder<D> device(D device) {
            this.device = device;
            return this;
        }

        public ExecutionCall<D> on(D device) {
            this.device = device;
            return build();
        }

        public Builder<D> operation(Operation operation) {
            this.operation = operation;
            return this;
        }

        public Builder<D> running(Operation operation) {
            this.operation = operation;
            return this;
        }

        public Builder<D> tensors(Tsr<?>... tensors) {
            this.tensors = tensors;
            return this;
        }

        public Builder<D> algorithm(Algorithm<?> algorithm) {
            this.algorithm = algorithm;
            return this;
        }

        public Builder<D> args(List<Arg> context) {
            this.context.addAll(context);
            return this;
        }

        public Builder<D> args( Arg<?>... context ) {
            return args(Arrays.stream(context).collect(Collectors.toList()));
        }

        public Builder<D> andArgs( List<Arg> context ) {
            this.context.addAll(context);
            return this;
        }

        public Builder<D> andArgs( Arg<?>... context ) {
            return args(Arrays.stream(context).collect(Collectors.toList()));
        }

        public ExecutionCall<D> build() {
            return new ExecutionCall<>(
                                device,
                                operation,
                                tensors,
                                algorithm,
                                context
                            );
        }
    }

    /**
     *  This is a simple nested class offering various lambda based methods
     *  for validating the tensor arguments stored inside this {@link ExecutionCall}.
     *  It is a useful tool readable as well as concise validation of a given
     *  request for execution, that is primarily used inside implementations of the middle
     *  layer of the backend-API architecture ({@link Algorithm#isSuitableFor(ExecutionCall)}).
     */
    
    public class Validator
    {
        private boolean _isValid = true;

        /**
         *  The validity as float being 1.0/true and 0.0/false.
         *
         * @return The current validity of this Validator as float value.
         */
        public float estimation() {
            return ( this._isValid ) ? 1.0f : 0.0f;
        }

        public Validator first( TensorCondition condition ) {
            if ( !condition.check( _tensors[ 0 ] ) ) this._isValid = false;
            return this;
        }

        public Validator any( TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr<?> t : _tensors ) any = condition.check( t ) || any;
            if ( !any ) this._isValid = false;
            return this;
        }

        public Validator anyNotNull( TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr<?> t : _tensors )
                if ( t != null ) any = condition.check( t ) || any;
            if ( !any ) this._isValid = false;
            return this;
        }

        public Validator all( TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr<?> t : _tensors ) all = condition.check( t ) && all;
            if ( !all ) this._isValid = false;
            return this;
        }

        public Validator allNotNull( TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr<?> t : _tensors )
                if ( t != null ) all = condition.check( t ) && all;
            if ( !all ) this._isValid = false;
            return this;
        }

        public Validator all( TensorCompare compare )
        {
            boolean all = true;
            Tsr<?> last = null;
            for ( Tsr<?> current : _tensors ) {
                if ( last != null && !compare.check( last, current ) ) all = false;
                last = current; // Note: shapes are cached!
            }
            if ( !all ) this._isValid = false;
            return this;
        }

        public Validator forDevice( DeviceCondition condition )
        {
            if ( !condition.check( _device ) ) this._isValid = false;
            return this;
        }

        public Validator forOperation( OperationCondition condition ) {
            if ( !condition.check(_operation) ) this._isValid = false;
            return this;
        }

        public boolean isValid() {
            return this._isValid;
        }
    }

}