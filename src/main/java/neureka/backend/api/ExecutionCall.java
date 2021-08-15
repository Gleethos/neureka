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
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.calculus.Function;
import neureka.devices.Device;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

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
     * This is an import property whose
     * role might not be clear at first :
     * An operation can have multiple inputs, however
     * when calculating the derivative for a forward or backward pass
     * then one must know which derivative ought to be calculated.
     * So the "derivative index" targets said input.
     * This property is -1 when no derivative should be calculated,
     * however 0... when targeting an input to calculate the derivative of.
     */
    private int _derivativeIndex = -1;

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
            int derivativeIndex,
            Operation operation,
            Tsr<?>[] tensors,
            int j,
            Algorithm<?> algorithm,
            List<Arg> context
    ) {
        this._device = device;
        this._derivativeIndex = derivativeIndex;
        /*
            This is an import property whose
            role might not be clear at first :
            An operation can have multiple inputs, however
            when calculating the derivative for a forward or backward pass
            then one must know which derivative ought to be calculated.
            So the "derivative index" targets said input.
            This property is -1 when no derivative should be calculated,
            however 0... when targeting an input to calculate the derivative of.
         */
        _arguments.set(Arg.DerivIdx.of(derivativeIndex));
        this._operation = operation;
        this._tensors = tensors;
        /*
            The following argument is relevant for a particular type of operation, namely: an "indexer". <br>
            An indexer automatically applies an operation on all inputs for a given function.
            The (indexer) function will execute the sub functions (of the AST) for every input index.
            If a particular index is not targeted however this variable will simply default to -1.
         */
        _arguments.set(Arg.VarIdx.of(j));
        this._algorithm = algorithm;
        for ( Arg<?> arg : context ) this.getMetaArgs().set(arg);
    }

    public static <DeviceType extends Device<?>> Builder<DeviceType> builder() { return new Builder<>(); }

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
        return this._derivativeIndex; // this.findAndGet(Arg.DerivIdx.class);
    }

    public Operation getOperation() {
        return this._operation;
    }

    public Tsr<?>[] getTensors() {
        return this._tensors;
    }

    public int getJ() {
        return this.getMetaArgs().getValOf(Arg.VarIdx.class);
    }

    public ExecutionCall<DeviceType> withTensors(Tsr<?>[] _tensors) {
        return this._tensors == _tensors
                ? this
                : new ExecutionCall<>(
                        this._device, this.getDerivativeIndex(), this._operation,
                        _tensors, this.getJ(), this._algorithm, _arguments.getAll(Arg.class)//this.getMetaArgs().findAll(Arg.class)
                    );
    }

    public ExecutionCall<DeviceType> withJ(int j) {
        return this.getJ() == j
                ? this
                : new ExecutionCall<>(
                        this._device, this.getDerivativeIndex(), this._operation,
                        this._tensors, j, this._algorithm, _arguments.getAll(Arg.class) //this.getMetaArgs().findAll(Arg.class)
                    );
    }


    public <V, T extends Arg<V>> V getValOf(Class<T> argumentClass ) {
        return _arguments.getValOf(argumentClass);
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
        return ExecutionCall.builder()
                                .device( newDevice )
                                .tensors( _tensors )
                                .operation( _operation )
                                .algorithm( _algorithm )
                                .args( _arguments.getAll(Arg.class) )//getMetaArgs().findAll(Arg.class) )
                                .args( Arg.DerivIdx.of( getDerivativeIndex() ) )
                                .build();
    }

    public <T extends Device<?>> ExecutionCall<T> forDeviceType(Class<T> type) {
        assert _device.getClass() == type;
        return (ExecutionCall<T>) this;
    }


    public <V> Tsr<V> getTsrOfType( Class<V> valueTypeClass, int i ) {
        // TODO: perform type checking!
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
        for ( Arg<?> arg : _arguments.getAll(Arg.class) ) call.getMetaArgs().set(arg);
        return getAlgorithm().supplyADAgentFor( function, call, forward );
    }

    public void mutateArguments( Mutator mutation ) {
        _tensors = mutation.mutate( _tensors );
    }

    public Validator validate() { return new Validator(); }

    public static class Builder<DeviceType extends Device<?>>
    {
        private DeviceType device;
        private int derivativeIndex = -1;
        private Operation operation;
        private Tsr<?>[] tensors;
        private int varIdx = -1;
        private Algorithm<?> algorithm;
        private final List<Arg> context = new ArrayList<>();

        Builder() { }

        public Builder<DeviceType> device(DeviceType device) {
            this.device = device;
            return this;
        }

        public Builder<DeviceType> derivativeIndex(int derivativeIndex) {
            this.derivativeIndex = derivativeIndex;
            return this;
        }

        public Builder<DeviceType> operation(Operation operation) {
            this.operation = operation;
            return this;
        }

        public Builder<DeviceType> tensors(Tsr<?>... tensors) {
            this.tensors = tensors;
            return this;
        }

        public Builder<DeviceType> j(int j) {
            this.varIdx = j;
            return this;
        }

        public Builder<DeviceType> algorithm(Algorithm<?> algorithm) {
            this.algorithm = algorithm;
            return this;
        }

        public Builder<DeviceType> args(List<Arg> context) {
            this.context.addAll(context);
            return this;
        }

        public Builder<DeviceType> args(Arg<?>... context) {
            return args(Arrays.stream(context).collect(Collectors.toList()));
        }

        public ExecutionCall<DeviceType> build() {
            return new ExecutionCall<>(
                            device,
                            derivativeIndex,
                            operation,
                            tensors,
                            varIdx,
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
            return ( _isValid ) ? 1.0f : 0.0f;
        }

        public Validator first( TensorCondition condition ) {
            if ( !condition.check( _tensors[ 0 ] ) ) _isValid = false;
            return this;
        }

        public Validator any( TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr<?> t : _tensors ) any = condition.check( t ) || any;
            if ( !any ) _isValid = false;
            return this;
        }

        public Validator anyNotNull( TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr<?> t : _tensors )
                if ( t != null ) any = condition.check( t ) || any;
            if ( !any ) _isValid = false;
            return this;
        }

        public Validator all( TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr<?> t : _tensors ) all = condition.check( t ) && all;
            if ( !all ) _isValid = false;
            return this;
        }

        public Validator allNotNull( TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr<?> t : _tensors )
                if ( t != null ) all = condition.check( t ) && all;
            if ( !all ) _isValid = false;
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
            if ( !all ) _isValid = false;
            return this;
        }

        public Validator forDevice( DeviceCondition condition )
        {
            if ( !condition.check( _device ) ) _isValid = false;
            return this;
        }

        public Validator forOperation( OperationCondition condition ) {
            if ( !condition.check(_operation) ) _isValid = false;
            return this;
        }

        public boolean isValid() {
            return this._isValid;
        }
    }

}