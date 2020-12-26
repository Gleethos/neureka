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

    A very simple class which wraps essential arguments and context data
    used for operation execution on Tsr instances.


*/

package neureka.backend.api;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.backend.api.algorithms.Algorithm;
import neureka.backend.api.operations.Operation;
import neureka.devices.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;

import java.util.Map;
import java.util.TreeMap;

/**
 * This class is a simple container holding relevant
 * arguments needed to execute on a targeted Device which
 * is specified by the type parameter below.
 *
 * It also holds a context map responsible for storing
 * operation specific variables.
 *
 * @param <DeviceType> The Device implementation targeted by an instance of this ExecutionCall!
 */
@Accessors( prefix = {"_"} )
@ToString
public class ExecutionCall< DeviceType extends Device >
{
    public interface TensorCondition { boolean check( Tsr tensor ); }
    public interface TensorCompare { boolean check( Tsr first, Tsr second ); }
    public interface DeviceCondition { boolean check( Device device ); }
    public interface OperationTypeCondition { boolean check( Operation type ); }
    public interface Mutator { Tsr[] mutate( Tsr[] tensors ); }

    @Getter
    private final DeviceType _device;
    /**
     * This method returns an import property whose
     * role might not be clear at first :
     * An operation can have multiple inputs, however
     * when calculating the derivative for a forward or backward pass
     * then one must know which derivative ought to be calculated.
     * So the "derivative index" targets said input.
     * This property is -1 when no derivative should be calculated,
     * however 0... when targeting an input to calculate the derivative of.
     *
     * @return The index of the input whose derivative ought to be calculated.
     */
    @Getter
    private final int _derivativeIndex;

    /**
     *  This is the operation type which will be applied to this execution call.
     *  It contains multiple implementations, one of which might be applicable to this call...
     */
    @Getter
    private final Operation _operation;

    @Getter
    private Tsr[] _tensors;

    @Getter
    private int _j = -1;

    private Algorithm<Algorithm> _implementation;

    @Getter
    private Map<String, Object> _context;

    @Accessors( prefix = {"_"} )
    public class Validator
    {
        @Getter
        private boolean _isValid = true;

        public float estimation() {
            return ( _isValid ) ? 1.0f : 0.0f;
        }

        public Validator first( TensorCondition condition ) {
            if ( !condition.check( _tensors[0] ) ) _isValid = false;
            return this;
        }

        public Validator any( TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr t : _tensors ) any = ( condition.check( t ) ) ? true : any;
            if ( !any ) _isValid = false;
            return this;
        }

        public Validator anyNotNull( TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr t : _tensors )
                if ( t != null ) any = ( condition.check( t ) ) ? true : any;
            if ( !any ) _isValid = false;
            return this;
        }

        public Validator all( TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr t : _tensors ) all = ( !condition.check( t ) ) ? false : all;
            if ( !all ) _isValid = false;
            return this;
        }

        public Validator allNotNull( TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr t : _tensors )
                if( t != null ) all = ( !condition.check( t ) ) ? false : all;
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

    public Validator validate() { return new Validator(); }

    public ExecutionCall(
            DeviceType device,
            Tsr[] tensors,
            int d,
            Operation type
    ) {
        _device = device;
        _tensors = tensors;
        _derivativeIndex = d;
        _operation = type;
        _implementation = null;
        _context = null;
    }
    
    public ExecutionCall(
            DeviceType device,
            Tsr[] tensors,
            int d,
            int j,
            Operation type
    ) {
        _device = device;
        _tensors = tensors;
        _derivativeIndex = d;
        _j = j;
        _operation = type;
        _implementation = null;
    }

    public Tsr getTensor( int i ) { return _tensors[ i ];}


    public Algorithm getImplementation() {
        if ( _implementation != null ) return _implementation;
        else _implementation = _operation.AlgorithmFor(this);
        return _implementation;
    }
    
    public boolean allowsForward() {
        return getImplementation().canAlgorithmPerformForwardADFor(this);
    }

    public boolean allowsBackward() {
        return getImplementation().canAlgorithmPerformBackwardADFor(this);
    }

    public ADAgent getADAgentFrom( Function function, ExecutionCall<Device> call, boolean forward )
    {
        if ( this._context != null ) {
            if ( call._context ==null ) call._context = new TreeMap<>();
            call._context.putAll(this._context);
        }
        return getImplementation().supplyADAgentFor(function, call, forward);
    }
    
    public void mutateArguments( Mutator mutation ) {
        _tensors = mutation.mutate(_tensors);
    }
    
    public ExecutionCall<DeviceType> withNew( Tsr[] tensors ) {
        return new ExecutionCall<DeviceType>(_device, tensors, _derivativeIndex, _j, _operation);
    }

    public ExecutionCall<DeviceType> withNew( DeviceType device ) {
        return new ExecutionCall<DeviceType>(device, _tensors, _derivativeIndex, _j, _operation);
    }

    public <T> T getAt( Class<T> type ) {
        if ( _context == null ) return null;
        return (T) _context.get(getClass().getName());
    }

    public Object getAt( String varName ) {
        if ( _context == null ) return null;
        return _context.get(varName);
    }

    public <T> ExecutionCall<DeviceType> putAt( String s, T o ) {
        if ( _context == null ) _context = new TreeMap<>();
        _context.put(s,o);
        return this;
    }

    public void takeContext( Map<String, Object>  context ) {
        if(_context==null && context!=null )_context = new TreeMap<>();
        if(context!=null) _context.putAll(_context);
    }

    // CONDITIONS:




}