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

import lombok.*;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.devices.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;

import java.util.Map;
import java.util.TreeMap;

/**
 *  This class is a simple container holding relevant
 *  arguments needed to execute on a targeted {@link Device} which
 *  is specified by the type parameter below. <br>
 *  <br>
 *  It also holds a context map responsible for storing
 *  operation specific variables.
 *
 * @param <DeviceType> The Device implementation targeted by an instance of this ExecutionCall!
 */
@Builder
@Accessors( prefix = {"_"} )
@ToString
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class ExecutionCall<DeviceType extends Device>
{
    public interface TensorCondition { boolean check( Tsr<?> tensor ); }
    public interface TensorCompare { boolean check( Tsr<?> first, Tsr<?> second ); }
    public interface DeviceCondition { boolean check( Device<?> device ); }
    public interface OperationTypeCondition { boolean check( Operation type ); }
    public interface Mutator { Tsr<?>[] mutate( Tsr<?>[] tensors ); }

    /**
     *  This field references the device on which this ExecutionCall should be executed.
     */
    @Getter @With // Generates a method which constructs a copy of this call with the provided device!
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
    @Getter
    @Builder.Default
    private int _derivativeIndex = -1;

    /**
     *  This is the operation type which will be applied to this execution call.
     *  It contains multiple implementations, one of which might be applicable to this call...
     */
    @Getter
    private final Operation _operation;

    /**
     *  The tensor arguments from which an operation will either
     *  read or to which it will write. <br>
     *  The first entry of this array is usually containing the output tensor,
     *  however this is not a necessity.
     *  Some operation algorithms might use multiple argument entries as output tensors.
     */
    @Getter @With // Generates a method which constructs a copy of this call with the provided tensors!
    private Tsr[] _tensors;

    /**
     *  The following parameter is relevant for a particular type of operation, namely: an "indexer". <br>
     *  An indexer automatically applies an operation on all inputs for a given function.
     *  The (indexer) function will execute the sub functions (of the AST) for every input index.
     *  If a particular index is not targeted however this variable will simply default to -1.
     */
    @Getter @With @Builder.Default
    private int _j = -1;

    /**
     *  This Algorithm variable is the chosen algorithm for a given execution call instance.
     *  The variable is initially null and will be chosen dynamically based on an access request
     *  to the corresponding getter method for this variable.
     *  So it is in essence a lazy load variable.
     *  Choosing an algorithm occurs through the {@link ExecutionCall#_operation} variable,
     *  which is of type {@link Operation} and contains multiple algorithms for different execution call scenarios...
     */
    @Builder.Default
    private Algorithm<?> _algorithm = null;

    /**
     *  Certain operations might require additionally parameters then the ones
     *  defined in this class... <br>
     */
    @Getter
    @Builder.Default
    private Map<String, Object> _context = null;

    private ExecutionCall(
            DeviceType device,
            Tsr<?>[] tensors,
            int d,
            Operation type,
            Algorithm<?> algorithm,
            Map<String, Object> context
    ) {
        _device = device;
        _tensors = tensors;
        _derivativeIndex = d;
        _operation = type;
        _algorithm = algorithm;
        _context = context;
    }


    public <T extends Device<?>> ExecutionCall<T> forDeviceType(Class<T> type) {
        assert _device.getClass() == type;
        return (ExecutionCall<T>) this;
    }

    public Tsr getTensor( int i ) { return _tensors[ i ];}


    public Algorithm<?> getAlgorithm() {
        if ( _algorithm != null ) return _algorithm;
        else _algorithm = _operation.getAlgorithmFor( this );
        return _algorithm;
    }

    public boolean allowsForward() {
        return getAlgorithm().canAlgorithmPerformForwardADFor( this );
    }

    public boolean allowsBackward() {
        return getAlgorithm().canAlgorithmPerformBackwardADFor( this );
    }

    public ADAgent getADAgentFrom( Function function, ExecutionCall<Device> call, boolean forward )
    {
        if ( this._context != null ) {
            if ( call._context == null ) call._context = new TreeMap<>();
            call._context.putAll( this._context );
        }
        return getAlgorithm().supplyADAgentFor( function, call, forward );
    }

    public void mutateArguments( Mutator mutation ) {
        _tensors = mutation.mutate( _tensors );
    }

    public <T> T getAt( Class<T> type ) {
        if ( _context == null ) return null;
        return (T) _context.get( type.getName() );
    }

    public Object getAt( String varName ) {
        if ( _context == null ) return null;
        return _context.get( varName );
    }

    public <T> ExecutionCall<DeviceType> putAt( String s, T o ) {
        if ( _context == null ) _context = new TreeMap<>();
        _context.put( s, o );
        return this;
    }

    public Validator validate() { return new Validator(); }

    /**
     *  This is a simple nested class offering various lambda based methods
     *  for validating the tensor arguments stored inside this {@link ExecutionCall}.
     *  It is a useful tool readable as well as concise validation of a given
     *  request for execution, that is primarily used inside implementations of the middle
     *  layer of the backend-API architecture ({@link Algorithm#isAlgorithmSuitableFor(ExecutionCall)}).
     */
    @Accessors( prefix = {"_"} )
    public class Validator
    {
        @Getter
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

        public Validator forOperation( OperationTypeCondition condition ) {
            if ( !condition.check(_operation) ) _isValid = false;
            return this;
        }
    }

}