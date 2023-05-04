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

 __________
 \__    ___\
    |  |____ _ __
    | /  ___/ '___\
    | \___  \ |
     \/_____/_|Impl      A long yet shallow class.

    This is the the core work-horse class of Neureka. The 'Tsr' class!
    It is a three-letter abbreviation of the word "Tensor"!

------------------------------------------------------------------------------------------------------------------------

   'Any fool can write code that a computer can understand.
    Good programmers write code that humans can understand.'
    – Martin Fowler

    Use the following as search keys :)

    §(1) : CONSTRUCTION
    §(2) : FLAGS
    §(3) : COMPONENT SYSTEM
    §(4) : PROPERTIES
    §(5) : OBJECT STATE MODIFICATION
    §(6) : ND-ITERATOR LOGIC
    §(7) : COMPONENT SPECIFIC
    §(8) : (OVERLOADABLE) OPERATORS & OPERATIONS
    §(9) : SLICING, INDEXING & INJECTING
    §(10) : MAPPING
*/

package neureka;

import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.LazyRef;
import neureka.backend.main.memory.MemUtil;
import neureka.backend.main.operations.other.ReLayout;
import neureka.common.composition.AbstractComponentOwner;
import neureka.common.composition.Component;
import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.fluent.slicing.SliceBuilder;
import neureka.fluent.slicing.SmartSlicer;
import neureka.fluent.slicing.states.AxisOrGetTsr;
import neureka.framing.NDFrame;
import neureka.framing.Relation;
import neureka.framing.fluent.AxisFrame;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.ndim.Filler;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterator.NDIterator;
import neureka.view.NdaAsString;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.*;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;


/**
 *  The implementation for the {@link Tsr} API.
 *
 * @param <V> The type parameter for the individual value items within this tensor.
 */
final class TsrImpl<V> extends AbstractNda<Tsr<V>, V> implements MutateTsr<V>
{
    static {
        _LOG = LoggerFactory.getLogger( TsrImpl.class );
    }

    /**
     *  This field contains multiple flags.
     *  The bits of this integer are used to encode various states which a tensor can have.
     *  These bits are flipped by bitmasks which are defined below.
     */
    private byte _flags = 0;

    /**
     *  The following fields are bit masks used to store true / false values
     *  in a targeted bit inside the {@link #_flags} variable.
     */
    private static final byte RQS_GRADIENT_MASK       = 1;
    private static final byte IS_VIRTUAL_MASK         = 2;
    private static final byte GRADIENT_APPLY_RQD_MASK = 4;
    private static final byte IS_DELETED_MASK         = 8;
    private static final byte IS_INTERMEDIATE_MASK    = 16;

    /*==================================================================================================================
    |
    |       §(1) : CONSTRUCTION
    |   ---------------------------
    */

    static <T> Tsr<T> _of( Object... args )
    {
        if ( args == null || args.length == 0 ) return new TsrImpl<>();
        if ( args.length == 1 ) {
            TsrImpl<T> t = new TsrImpl<>();
            boolean success = constructFor(t, CPU.get(), NDConstructor.of(1)).newPopulatedFromOne( args[ 0 ], args[ 0 ].getClass() );
            if ( !success ) {
                String message = "Cannot create tensor from argument of type '" + args[ 0 ].getClass().getName() + "'!";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
            return t;
        }
        args[ 0 ] = ( args[ 0 ] instanceof List ) ? ( (List<?>) args[ 0 ] ).toArray() : args[ 0 ];
        args[ 1 ] = ( args[ 1 ] instanceof List ) ? ( (List<?>) args[ 1 ] ).toArray() : args[ 1 ];

        Class<?> commonType = _extractCommonType(args);
        if ( commonType != null ) {
            TsrImpl<T> t = new TsrImpl<>();
            constructFor(t, CPU.get(), NDConstructor.of( args.length ))
                .tryConstructing(
                    DataType.of(commonType),
                    args
                );
            return t;
        }

        /* EXPRESSION BASED CONSTRUCTION:
            The following allows the creation of tensors based on passing an expression
            alongside input tensors to the constructor.
            An example would be:

                Tsr<?> t = Tsr.of( "tanh(", x, ") * 7 **", y );
        */
        boolean containsString = false;
        int numberOfTensors = 0;
        for ( Object o : args ) {
            containsString = ( o instanceof String ) || containsString;
            if ( o instanceof TsrImpl)
                numberOfTensors++;
        }
        TsrImpl<T>[] tensors = new TsrImpl[ numberOfTensors ];
        StringBuilder f = new StringBuilder();
        int ti = 0;
        for ( Object o : args ) {
            if ( o instanceof TsrImpl) {
                tensors[ ti ] = ( (TsrImpl<T>) o );
                f.append( "I[" ).append( ti ).append( "]" );
                ti++;
            }
            else if ( o instanceof String ) f.append( (String) o );
            else
                _LOG.debug(
                    "Unexpected tensor construction argument of type '"+o.getClass().getSimpleName()+"'"
                );
        }
        if ( tensors.length == 0 || tensors[0] == null) return new TsrImpl<>();
        return Function.of( f.toString(), true ).call( tensors );
    }

    static <T> Tsr<T> _of( Iterable<T> iterable )
    {
        TsrImpl<T> t = new TsrImpl<>();
        List<T> list = new ArrayList<>();
        iterable.forEach( list::add );
        return _of( t );
    }

    static <T> Tsr<T> _of( List<T> list )
    {
        TsrImpl<T> t = new TsrImpl<>();
        Class<?> commonType = _extractCommonType( list.toArray() );
        // We construct the tensor:
        constructFor(t, CPU.get(), NDConstructor.of( list.size() ))
                    .tryConstructing(
                        DataType.of(commonType),
                        list.toArray()
                    );
        return t;
    }


    /**
     * @param args The objects which should be checked.
     * @return A common type or null if they are not all of the same type.
     */
    private static Class<?> _extractCommonType( Object... args ) {
        Class<?> commonType = null;
        for ( Object o : args )
            if ( o != null ) {
                if ( commonType == null ) commonType = o.getClass();
                else if ( !commonType.equals(o.getClass()) ) return null;
            }

        return commonType;
    }

    // Constructors:

    /**
     *  This constructor creates a completely empty tensor which is void of any contents and meaning.
     *  The use case for this would be to use the produced {@link Tsr}
     *  instance as a target for an inline operation which fills this instance with an actual value. <br>
     *  An example of this approach would be to call the {@link #putAt(List, Nda)} method with an empty list as key.
     *  This will be interpreted as an inline copy of the contents of the
     *  second parameter into this {@link Tsr} instance.
     *  This constructor will be called by the {@link Tsr#newInstance()} factory method.
     */
    TsrImpl() {
        _setData(new Data<V>() {
            @Override public Device<V> owner() { return (Device<V>) CPU.get(); }
            @Override public Object getOrNull() { return null;}
            @Override public DataType<V> dataType() {
                return (DataType<V>) Neureka.get().settings().dtype().getDefaultDataType();
            }

            @Override public int usages() { return 1; }
        });
    }

    public static <V> TsrImpl<V> _of( NDConstructor ndConstructor, Device device, DataType<V> dataType, Object value ) {
        Object data = value;
        if ( List.class.isAssignableFrom( dataType.getItemTypeClass() ) )
            data = new Object[]{ value }; // Make an nd-array of lists possible"
        if ( Object[].class.isAssignableFrom( dataType.getItemTypeClass() ) )
            data = new Object[]{ value }; // Make an nd-array of arrays possible"
        if ( Object.class == dataType.getItemTypeClass() ) {
            if ( value.getClass() != Object[].class )
                data = new Object[]{ value };
        }
        if ( data instanceof List<?> ) {
            List<?> range = (List<?>) data;
            data = range.toArray();// TODO: This is probably wrong!
        }
        TsrImpl<V> t = new TsrImpl<>();
        constructFor(t, device, ndConstructor).tryConstructing( dataType, data );
        return t;
    }

    static <V> TsrImpl<V> _of( NDConstructor ndConstructor, DataType<V> dataType, Data<V> data ) {
        // We check if the type of the data is compatible with the type of the tensor:
        if ( !dataType.getItemTypeClass().isAssignableFrom( data.dataType().getItemTypeClass() ) )
            throw new IllegalArgumentException(
                    "The data type of the data is not compatible with the data type of the tensor!"
                );

        TsrImpl<V> t = new TsrImpl<>();
        constructFor(t, data.owner(), ndConstructor).constructTrusted( data );
        return t;
    }

    /**
     *  see {@link Tsr#of(DataType, Shape, Filler)}
     */
    static <V> TsrImpl<V> _of( NDConstructor ndConstructor, DataType<V> type, Filler<V> filler ) {
        LogUtil.nullArgCheck(ndConstructor, "ndcProducer", NDConstructor.class );
        LogUtil.nullArgCheck( type, "type", DataType.class );
        LogUtil.nullArgCheck( type, "filler", Filler.class );
        TsrImpl<V> t = new TsrImpl<>();
        constructFor(t, CPU.get(), ndConstructor).unpopulated( false, true, type );
        t._initDataArrayFrom( filler );
        return t;
    }

    /**
     *  See {@link Tsr#of(Class, Shape, neureka.math.args.Arg.Seed)} and {@link #of(List, String)}
     */
    static <V> TsrImpl<V> _of( Class<V> valueType, NDConstructor ndConstructor, Arg.Seed seed ) {
        LogUtil.nullArgCheck( valueType, "valueType", Class.class );
        LogUtil.nullArgCheck(ndConstructor, "ndcProducer", NDConstructor.class );
        LogUtil.nullArgCheck( seed, "seed", Arg.Seed.class );
        TsrImpl<V> t = new TsrImpl<>();
        constructFor(t, CPU.get(), ndConstructor).newSeeded( valueType, seed );
        return t;
    }

    static <V> TsrImpl<V> _of( NDConstructor ndConstructor, DataType<?> type ) {
        LogUtil.nullArgCheck(ndConstructor, "ndcProducer", NDConstructor.class );
        LogUtil.nullArgCheck( type, "type", DataType.class );
        TsrImpl<V> t = new TsrImpl<>();
        constructFor(t, CPU.get(), ndConstructor).unpopulated( true, true, type );
        return t;
    }

    /*==================================================================================================================
    |
    |       §(2) : FLAGS
    |   ----------------------
    */

    /** {@inheritDoc} */
    @Override
    public Tsr<V> setRqsGradient(boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient ) {
            if ( !rqsGradient ) this.remove( TsrImpl.class );
            else if ( has(GraphNode.class) ) {
                if ( getGraphNode().map( n -> n.getMode() == 0 ).orElse(false) )
                    remove(GraphNode.class);
                else
                    throw new IllegalArgumentException(
                        "This tensor is already part of a gradient dependent graph as " +
                        "branch node and therefore cannot be removed from it."
                    );
            }
        }
        _setRqsGradient( rqsGradient );
        return this;
    }

    /** {@inheritDoc} */
    @Override public boolean rqsGradient() { return ( _flags & RQS_GRADIENT_MASK ) == RQS_GRADIENT_MASK; }

    private void _setRqsGradient(boolean rqsGradient) {
        if ( rqsGradient() != rqsGradient ) {
            if ( rqsGradient ) _flags += RQS_GRADIENT_MASK;
            else               _flags -= RQS_GRADIENT_MASK;
        }
    }

    /** {@inheritDoc} */
    @Override public boolean isIntermediate() { return ( _flags & IS_INTERMEDIATE_MASK ) == IS_INTERMEDIATE_MASK; }

    /**
     *  Intermediate tensors are internal non-user tensors which may be eligible
     *  for deletion when further consumed by a {@link Function}.
     *  For the casual user of Neureka, this flag should always be false!
     *
     * @param isIntermediate The truth value determining if this tensor is not a user tensor but an internal
     *                       tensor which may be eligible for deletion by {@link Function}s consuming it.
     */
    private Tsr<V> _setIsIntermediate(boolean isIntermediate) {
        if ( isIntermediate() != isIntermediate ) {
            if ( isIntermediate ) _flags += IS_INTERMEDIATE_MASK;
            else                  _flags -= IS_INTERMEDIATE_MASK;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isVirtual() { return ( _flags & IS_VIRTUAL_MASK ) == IS_VIRTUAL_MASK; }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> setIsVirtual( boolean isVirtual )
    {
        if ( getNDConf() == null )
            throw new IllegalStateException(
                "Cannot set the virtual flag of a tensor which has not been constructed yet!"
            );

        if ( isVirtual() != isVirtual )
        {
            if ( isVirtual )
                _virtualize();
            else
                _actualize();
            // Virtual and actual tensors require a different mapping from a given index to the underlying data..
            // Therefore, we need to re-initialize the NDConfiguration object:
            constructFor(this, getDevice(),NDConstructor.of(getNDConf().shape())).unpopulated( isVirtual, false, getDataType() );
            if ( isVirtual )
                this.find( Relation.class )
                        .ifPresent( r ->
                            r.getChildren().forEach(c -> {
                                ((TsrImpl<V>)c)._setData( _getData() );
                                ((TsrImpl<V>)c).setIsVirtual( true );
                            })
                        );
            else
                this.find(Relation.class)
                    .map( relation -> ((Relation<V>)relation).getParent().orElse(null) )
                    .map( parent -> parent.get(Relation.class) )
                    .ifPresent( parentRelation -> parentRelation.removeChild( this ) );
        }
        else if ( isVirtual ) _allocateVirtual(); //> Only a single value representing the rest.
        return this;
    }

    /**
     *  This method is the inner counterpart to the public "{@link MutateTsr#setIsVirtual}" method.
     *  It actually performs the bit flipping by applying the corresponding bit mask. <br>
     *  <br>
     * @param isVirtual The truth value which ought to be applied.
     */
    @Override
    protected void _setIsVirtual( boolean isVirtual ) {
        if ( isVirtual() != isVirtual ) {
            if ( isVirtual ) _flags += IS_VIRTUAL_MASK;
            else             _flags -= IS_VIRTUAL_MASK;
        }
    }

    /**  {@inheritDoc} */
    @Override
    public boolean isDeleted() { return ( _flags & IS_DELETED_MASK ) == IS_DELETED_MASK; }

    /** {@inheritDoc} */
    @Override
    public boolean gradientApplyRequested() { return ( _flags & GRADIENT_APPLY_RQD_MASK ) == GRADIENT_APPLY_RQD_MASK; }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> setGradientApplyRequested(boolean applyRequested ) {
        if ( gradientApplyRequested() != applyRequested ) {
            if ( applyRequested ) {
                if (
                    Neureka.get().settings().autograd().isApplyingGradientWhenRequested() &&
                    !Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed()
                )
                    this.applyGradient();
                else
                    _flags += GRADIENT_APPLY_RQD_MASK;
            }
            else _flags -= GRADIENT_APPLY_RQD_MASK;
        }
        return this;
    }

    /**
     *  Although tensors will be garbage collected when they are not strongly referenced,
     *  there is also the option to manually free up the tensor and its associated data.
     *  This is especially useful when tensors are stored on a device like the OpenCLDevice.
     *  In that case calling the "{@link MutateTsr#delete()}" method will free the memory reserved for this tensor.
     *  This manual memory freeing through this method can be faster than waiting for
     *  the garbage collector to kick in... <br>
     *  <br>
     *
     * @return This very tensor instance to allow for method chaining.
     */
    private Tsr<V> _delete()
    {
        if ( isDeleted() ) return this;
        getGraphNode().ifPresent( n -> {
            if ( !n.canBeDeleted() ) {
                String message = "Cannot delete a tensor which is used as derivative by the AD computation graph!";
                _LOG.error( message );
                throw new IllegalStateException( message );
            }
        });
        this.find( Device.class ).ifPresent( device -> device.free( this ) );
        _setData( null );
        _setNDConf( null );
        _flags = 0;
        this.find( TsrImpl.class ).ifPresent( t -> t.mut().delete() );
        _deleteComponents();
        _flags += IS_DELETED_MASK;
        return this;
    }

    /*==================================================================================================================
    |
    |       §(3) : COMPONENT SYSTEM
    |   --------------------------------
    */

    /** {@inheritDoc} */
    @Override public <T extends Component<?>> T get( Class<T> componentClass )
    {
        LogUtil.nullArgCheck( componentClass, "componentClass", Class.class );

        if ( GraphNode.class.isAssignableFrom(componentClass) )
            _guardGet(componentClass.getSimpleName());
        else if ( NDFrame.class.isAssignableFrom(componentClass) )
            _guardGet(componentClass.getSimpleName());

        return super.get(componentClass);
    }

    /**
     * This method is executed when a new Component is added to the tensor.
     * The public add method is implemented in the super class
     * '{@link AbstractComponentOwner}' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. ({@link GraphNode}, {@link NDFrame}, {@link Relation}, int[], ...)
     * @return The unchanged object or maybe in future versions: null (component rejected)
     */
    @Override
    protected < T extends Component<Tsr<V>> > T _setOrReject(T newComponent ) { return newComponent; }

    /**
     * This method is executed when a component is being removed from the tensor.
     * The public remove method is implemented in the super class
     * '{@link AbstractComponentOwner}' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. ({@link GraphNode}, {@link NDFrame}, {@link Relation}, int[], ...)
     * @return The unchanged object or when rejected: null (component rejected)
     */
    @Override
    protected <T extends Component<Tsr<V>>> T _removeOrReject(T newComponent )
    {
        if ( newComponent instanceof Device ) {
            Device<V> device = (Device<V>) newComponent;
            /*
                The following seems like a redundant check, however often times a tensor
                will be removed from a Device implementation inside the "restore" method
                when the tensor has already been removed from the device...
                Without the condition below a stack overflow would occur!
             */
            if ( device.has( this ) ) {
                try {
                    device.restore( this );
                } catch ( Exception exception ) {
                    _LOG.error(
                        "Removing device from tensor / tensor from device failed.\n" +
                        "Restoring tensor from device threw exception.\n",
                        exception
                    );
                    throw exception;
                }
            }
        }
        return newComponent;
    }


    /*==================================================================================================================
    |
    |       §(4) : PROPERTIES :
    |   ---------------------------------------
    */

    /**
     *  {@inheritDoc}
     */
    @Override
    public int getVersion() { return _version; }


    /*==================================================================================================================
    |
    |       §(5) : OBJECT STATE MODIFICATION :
    |   ------------------------------------------
    */

    /**
     * This method is responsible for incrementing
     * the "_version" field variable which represents the version of the data of this tensor.
     * Meaning :
     * Every time the underlying data (_value) changes this version ought to increment alongside.
     * The method is called during the execution procedure.
     *
     * @param call The context object containing all relevant information that defines a call for tensor execution.
     */
    private void _incrementVersionBecauseOf( ExecutionCall<?> call ) {
        if ( Neureka.get().settings().autograd().isPreventingInlineOperations() ) {
            _version++; // Autograd must be warned!
            GraphNode<?> node = get( GraphNode.class );
            if ( node != null && node.getPayloadReferenceVersion() != _version ) {
                if ( node.usesAD() || node.isUsedAsDerivative() ) {
                    String error = "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\n" +
                                   "The following OperationType caused an internal version mismatch: '"+call.getOperation().getIdentifier()+"'";
                    _LOG.error( error );
                    throw new IllegalStateException( error );
                }
            }
        }
    }

    /**
     *  In essence tensors are merely fancy wrapper for some form of array of any type... 
     *  This wrapper usually stays the same of a given data array.
     *  However, sometimes a tensor changes its identity, or rather the underlying
     *  data changes the wrapping tensor instance. <br>
     *  <br>
     * @param tensor The tensor whose identity should be stolen.
     */
    private void _become( TsrImpl<V> tensor )
    {
        if ( tensor == null ) return;
        _setData( tensor.getMut().getData() );
        _setNDConf( tensor.getNDConf() );
        _flags = tensor._flags;
        _transferFrom( tensor );
        tensor._setData( null );
        tensor._setNDConf( null );
        tensor._flags = 0;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public MutateTsr<V> getMut() {
        _guardGet("unsafe API");
        return this;
    }

    /** {@inheritDoc} */
    @Override public MutateNda.Item<V> at(int... indices ) {
        return new MutateNda.Item<V>() {
            @Override public V orElseNull() { return item( indices ); }
            @Override public void set( V value ) { getMut().putAt( indices, value ); }
            @Override public boolean equals( Object o ) {
                if ( o == null ) return false;
                if ( o == this ) return true;
                if ( o.getClass() != this.getClass() ) return false;
                Nda.Item<V> other = (Nda.Item<V>) o;
                return this.get().equals( other.get() );
            }
            @Override public int hashCode() { V item = get(); return ( item == null ? 0 : item.hashCode() ); }
            @Override public String toString() { return String.valueOf( get() ); }
        };
    }
    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setNDConf(NDConfiguration configuration ) { TsrImpl.this._setNDConf( configuration ); return TsrImpl.this; }

    /**
     *  {@inheritDoc}
     */
    @Override
    public <V> Tsr<V> toType( Class<V> typeClass ) {
        LogUtil.nullArgCheck( typeClass, "typeClass", Class.class, "Cannot convert tensor to 'null' data type." );
        return TsrImpl.this._toType( typeClass );
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public <U> Tsr<U> upcast( Class<U> superType ) {
        LogUtil.nullArgCheck( superType, "superType", Class.class );
        if ( superType.isAssignableFrom(TsrImpl.this.itemType()) )
            return (Tsr<U>) TsrImpl.this;
        else
            throw new IllegalArgumentException("Provided type '"+superType+"' is not a super type of '"+ TsrImpl.this.itemType()+"'.");
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> toLayout( NDConfiguration.Layout layout ) {
        ReLayout.toLayout( this, layout );
        return TsrImpl.this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> incrementVersion( ExecutionCall<?> call ) {
        LogUtil.nullArgCheck( call, "call", ExecutionCall.class );
        _incrementVersionBecauseOf( call );
        return TsrImpl.this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setIsIntermediate( boolean isIntermediate ) { return _setIsIntermediate( isIntermediate ); }

    /**
     *  {@inheritDoc}
     */
    @Override public Tsr<V> delete() { return TsrImpl.this._delete(); }

    /**
     *  {@inheritDoc}
     */
    @Override public Data<V> getData() { return _getData(); }

    /**
     *  {@inheritDoc}
     */
    @Override
    public <A> A getDataAs( Class<A> arrayTypeClass ) {
        return DataConverter.get().convert( _getData(false), arrayTypeClass );
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setDataAt( int i, V o ) {
        _guardMod("data object");
        _setDataAt( i, o );
        return TsrImpl.this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setData( Data<V> data ) {
        TsrImpl.this._setData( data );
        return TsrImpl.this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override public Tsr<V> detach() { TsrImpl.this.remove( GraphNode.class ); return TsrImpl.this; }

    /** {@inheritDoc} */
    @Override public Tsr<V> timesAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot multiply-assign 'null' to a tensor!");
        return Neureka.get().backend().getFunction().mulAssign().call( TsrImpl.this, other );
    }

    /** {@inheritDoc} */
    @Override public Tsr<V> timesAssign( V other ) {
        LogUtil.nullArgCheck(other, "other", TsrImpl.this.getItemType(), "Cannot multiply-assign 'null' to a tensor!");
        return this.timesAssign( Tsr.of( getItemType(), this.shape(), other ) );
    }

    /** {@inheritDoc} */
    @Override public Tsr<V> divAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot divide-assign a tensor by 'null' (In any sense of the word)!");
        return Neureka.get().backend().getFunction().divAssign().call( TsrImpl.this, other );
    }

    /** {@inheritDoc} */
    @Override public Tsr<V> modAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform tensor modulo 'null'!");
        return Neureka.get().backend().getFunction().modAssign().call( TsrImpl.this, other );
    }

    /** {@inheritDoc} */
    @Override public Tsr<V> plusAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot add-assign 'null' to a tensor!");
        return Neureka.get().backend().getFunction().plusAssign().call( TsrImpl.this, other );
    }

    /** {@inheritDoc} */
    @Override public Tsr<V> minusAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot subtract-assign 'null' from a tensor!");
        return Neureka.get().backend().getFunction().minusAssign().call( TsrImpl.this, other );
    }

    /** {@inheritDoc} */
    @Override public Tsr<V> minusAssign( V other ) {
        LogUtil.nullArgCheck(other, "other", TsrImpl.this.getItemType(), "Cannot subtract-assign 'null' from a tensor!");
        return minusAssign(
                Tsr.of( TsrImpl.this.getDataType().getItemTypeClass() )
                        .withShape(TsrImpl.this.getNDConf().shape())
                        .all(other)
        );
    }

    @Override
    public Tsr<V> assign( V other ) {
        LogUtil.nullArgCheck(other, "other", TsrImpl.this.getItemType(), "Cannot subtract-assign 'null' from a tensor!");
        return assign(
                Tsr.of( TsrImpl.this.getDataType().getItemTypeClass() )
                        .withShape(TsrImpl.this.getNDConf().shape())
                        .all(other)
        );
    }

    @Override
    public Tsr<V> assign( Nda<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot assign 'null' to a tensor!");
        return Neureka.get().backend().getFunction().idy().call( TsrImpl.this, (Tsr<V>) other );
    }

    @Override
    public Tsr<V> labelAxes( String[]... labels ) {
        LogUtil.nullArgCheck(labels, "labels", String[][].class, "Tensors cannot be labeled 'null'!");
        if ( labels.length > this.rank() )
            throw new IllegalArgumentException(
                    "Number of the provided axes labels is larger than the total number of axes (rank) of the nd-array."
            );

        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) {
            frame = new NDFrame<>( this, null);
            this.set(frame);
        }
        for ( int i = 0; i < labels.length; i++ ) {
            if ( labels[ i ] != null ) {
                AxisFrame<Integer, V> atAxis = frame.atAxis( i );
                for ( int ii = 0; ii < labels[ i ].length; ii++ ) {
                    if ( labels[ i ][ ii ] != null )
                        atAxis.atIndexAlias( labels[ i ][ ii ] ).setIndex( ii );
                }
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> labelAxes( List<List<Object>> labels ) {
        LogUtil.nullArgCheck(labels, "labels", List.class, "Tensors cannot be labeled 'null'!");
        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) set( new NDFrame<>( labels, this, null ) );
        else set( frame.withAxesLabels( labels ) );
        return TsrImpl.this;
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> label( String label ) {
        LogUtil.nullArgCheck( label, "label", List.class, "Tensors cannot be labeled 'null'!" );
        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) set( new NDFrame<>( Collections.emptyList(), this, label ) );
        else set( frame.withLabel(label) );
        return TsrImpl.this;
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> labelAxes( Map<Object, List<Object>> labels )
    {
        LogUtil.nullArgCheck(labels, "labels", Map.class, "Tensors cannot be labeled 'null'!");
        String label = getLabel();
        label = label == null || label.isEmpty() ? null : label;
        TsrImpl.this.set( new NDFrame<>( labels, TsrImpl.this, label ) );
        return TsrImpl.this;
    }

    /*==================================================================================================================
    |
    |       §(6) : ND-ITERATOR LOGIC :
    |   ---------------------------------------
    */

    /**
     * This method returns an iterator over the elements of this tensor. <br>
     *
     * @return An iterator over elements of type ValType.
     */

    @Override
    public Iterator<V> iterator()
    {
        NDIterator _ndi = NDIterator.of( this );
        return new Iterator<V>()
        {
            private final int _size = TsrImpl.this.size();
            private int _count = 0;

            @Override public boolean hasNext() { return _count != _size; }

            @Override
            public V next() {
                V value = TsrImpl.this.getDataAt( _ndi.i() );
                _ndi.increment();
                _count ++;
                return value;
            }
        };
    }


    /*==================================================================================================================
    |
    |       §(7) : COMPONENT SPECIFIC :
    |   ---------------------------------------
    */

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> to( Device<?> device ){
        if ( this.getDevice() != device ) super._set( device );
        return this;
    }

    /**
     * @param error A lazy reference to a supplier of the error tensor which
     *              may not be called if the error is not needed.
     *              This is to avoid unnecessary allocations and computations.
     */
    void _backward( LazyRef<Tsr<V>> error ) {
        LogUtil.nullArgCheck(error, "error", Tsr.class, "Cannot back-propagate 'null'!");
        LazyRef<Tsr<V>> errorRef = this.isOutsourced()
                                      ? LazyRef.of(()->error.get().deepCopy().to(this.getDevice()))
                                      : error;

        find( GraphNode.class ).ifPresent( node -> node.backward(errorRef.get()) );

        if ( this.rqsGradient() )
            mut().addToGradient( errorRef.get() );
    }

    @Override
    public Tsr<V> withLabel( String label ) {
        Tsr<V> copy = this.shallowCopy();
        if ( copy.label().endsWith(":slice") ) // We remove the slice postfix if it exists...
            copy = copy.shallowClone().mut().label( copy.label().substring(0, copy.label().length()-6) );
        return copy.mut().label( label );
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> withLabels( String[]... labels ) {
        Tsr<V> copy = this.shallowCopy();
        if ( copy.label().endsWith(":slice") ) // We remove the slice postfix if it exists...
            copy = copy.shallowClone().mut().label( copy.label().substring(0, copy.label().length()-6) );
        return copy.mut().labelAxes( labels );
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> withLabels( List<List<Object>> labels ) {
        Tsr<V> copy = this.shallowCopy();
        if ( copy.label().endsWith(":slice") ) // We remove the slice postfix if it exists...
            copy = copy.shallowClone().mut().label( copy.label().substring(0, copy.label().length()-6) );
        return copy.getMut().labelAxes( labels );
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> withLabels( Map<Object, List<Object>> labels ) {
        Tsr<V> copy = this.shallowCopy();
        if ( copy.label().endsWith(":slice") ) // We remove the slice postfix if it exists...
            copy = copy.shallowClone().mut().label( copy.label().substring(0, copy.label().length()-6) );
        return copy.getMut().labelAxes( labels );
    }

    /*==================================================================================================================
    |
    |       §(8) : (OVERLOADABLE) OPERATORS & OPERATIONS :
    |   -----------------------------------------------------
    |       ...for more context see package 'math'...
    |*/

    /** {@inheritDoc} */
    @Override
    public boolean isCase( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform 'is case' operation when second operand is 'null'!");
        return this.find( Relation.class )
                    .map( r -> ((Relation<?>)r).getChildren().stream().anyMatch( (Tsr<?> c) -> c.equals(other) ))
                    .orElse(false);
    }

    /*==================================================================================================================
    |
    |       §(9) : SLICING, INDEXING & INJECTING :
    |   -----------------------------------------------------
    |       ...for more context see package 'ndim.config'...
    */

    /** {@inheritDoc} */
    @Override
    public Tsr<V> getAt( int... indices ) {
        LogUtil.nullArgCheck(indices, "indices", int[].class, "Indices array must not be 'null'!");
        return getAt( Arrays.stream( indices ).boxed().toArray() );
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> getAt( Map<?,Integer> rangToSteps) {
        LogUtil.nullArgCheck(rangToSteps, "rankToSteps", Map.class, "Rank-to-steps map must not be 'null'!");
        // ...not a simple slice... Advanced:
        return SmartSlicer.slice(new Object[]{rangToSteps}, this);
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> getAt( List<?> key ) {
        LogUtil.nullArgCheck( key, "key", List.class );
        if ( key.stream().anyMatch( i -> i == null ) )
            throw new IllegalArgumentException("List of indices/ranges may not contain entries which are null!");
        if ( key.isEmpty() ) {
            /*
                An empty List instance is being interpreted as
                the request to create an identical slice, meaning that the
                resulting tensor views the same data as its parent while not
                being the same instance. (In a sense, its a shallow copy!)
             */
            return shallowCopy();
        }

        Object[] indices = key.toArray();

        boolean allInt = true;
        for ( Object o : indices ) allInt = allInt && o instanceof Integer;
        if ( allInt && indices.length == rank() ) {
            int[] newOffset = DataConverter.get().convert(indices, int[].class);
            for ( int i = 0; i < this.rank(); i++ )
                newOffset[ i ] = ( newOffset[ i ] < 0 ) ? getNDConf().shape( i ) + newOffset[ i ] : newOffset[ i ];
            for ( int i = 0; i < this.rank(); i++ )
                indices[ i ] = newOffset[ i ];
            allInt = false;
        }
        boolean hasScale = false;
        for ( Object o : indices ) hasScale = hasScale || o instanceof Map;
        return SmartSlicer.slice(
                ( allInt ? new Object[]{ DataConverter.get().convert(indices, int[].class) } : indices ),
                this
            );
    }

    /** {@inheritDoc} */
     @Override
    public TsrImpl<V> deepCopy() {
         return _clone( false );
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> deepClone() {
        return _clone( true );
    }

    private TsrImpl<V> _clone(boolean autograd) {
        Function cloner = autograd ? Neureka.get().backend().getAutogradFunction().idy() : Neureka.get().backend().getFunction().idy();
        boolean thisIsIntermediate = this.isIntermediate();
        _setIsIntermediate( false );
        Tsr<V> clone = Tsr.like( this )
                .all( (V) Double.valueOf(0.0) );

        if ( clone.itemType() != this.itemType() )
            throw new IllegalStateException("Item type of clone must be the same as the item type of the original!");

        clone = cloner.call( clone, this );
        clone.getMut().setIsIntermediate( thisIsIntermediate );
        _setIsIntermediate( thisIsIntermediate );
        return (TsrImpl<V>) clone;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public AxisOrGetTsr<V> slice() { return new SliceBuilder<>( this ); }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> putAt( List<?> key, Nda<V> value ) {
        _putAtCheckFor( (Tsr<?>) value );
        Tsr<V> slice = ( key == null ) ? this : getAt( key );
        Data<V> thisData = this.getMut().getData();
        Object thisDataRef = ( thisData != null ? thisData.getOrNull() : null );
        if ( thisDataRef != null && !thisDataRef.equals(slice.getMut().getData().getOrNull()) )
            throw new IllegalStateException("Failed to isolate slice for inline assignment!");

        return _putAt( slice, (Tsr<V>) value );
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> putAt( int[] indices, V item ) {
        if ( indices == null )
            throw new IllegalArgumentException( "Provided indices are null!" );
        if ( indices.length > this.rank() ) {
            int[] correct = new int[rank()];
            System.arraycopy( indices, 0, correct, 0, indices.length );
            indices = correct;
        } else if ( indices.length < rank() ) {
            int[] correct = new int[rank()];
            System.arraycopy( indices, 0, correct, 0, indices.length );
            for ( int i = indices.length; i < rank(); i++ ) correct[i] = 0;
            indices = correct;
        }
        if ( this.isVirtual() && this.size() > 1 ) this.setIsVirtual( false );
        int i = getNDConf().indexOfIndices(indices);
        this.getMut().setDataAt( i, item );
        return this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> putAt( Map<?,Integer> key, Nda<V> value ) {
        _putAtCheckFor((Tsr<?>) value);
        Tsr<V> slice = ( key == null ) ? this : getAt( key );
        return _putAt( slice, (Tsr<V>) value);
    }

    private void _putAtCheckFor( Tsr<?> value ) {
        if ( value.isEmpty() ) {
            String message = "Provided tensor is empty! Empty tensors cannot be injected.";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
    }

    private Tsr<V> _putAt( Tsr<V> slice, Tsr<V> value )
    {
        boolean valueIsDeviceVisitor = false;
        if ( slice.isOutsourced() && !value.isOutsourced() ) {
            Device<V> device = slice.getDevice();
            try {
                device.store( value );
            } catch ( Exception e ) {
                _LOG.error( "Trying to migrate target slice tensor to device failed.", e );
                throw e;
            }
            valueIsDeviceVisitor = true;
        }
        if ( this.isEmpty() && slice.isEmpty() || slice.size() != value.size() ) _become((TsrImpl<V>) value); // TODO: Rethink this a little
        else Neureka.get().backend().getFunction().idy().call(  slice, value  );
        try {
            if ( valueIsDeviceVisitor ) value.getDevice().restore( value );
        } catch ( Exception exception ) {
            _LOG.error( "Trying to migrate source tensor back to original location failed.", exception );
            throw exception;
        }
        return this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public V getDataAt( int i ) { return getDevice().access( this ).readAt( i ); }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setItemAt( int i, V o ) {
        _guardMod("data object");
        NDConfiguration ndc = this.getNDConf();
        _setDataAt( ndc.indexOfIndex( i ), o );
        return this;
    }

    /** {@inheritDoc} */
    @Override public Tsr<V> putAt( List<?> indices, V value ) {
        if ( indices.stream().allMatch( i -> i instanceof Number ) )
            return setItemAt( indexOfIndices(indices.stream().mapToInt( i -> ((Number)i).intValue() ).toArray()), value );
        else
            return this.putAt( indices, Tsr.ofAny( this.getItemType(), shape(), value ) );
    }

    /** {@inheritDoc} */
    @Override public Tsr<V> putAt( int index, V value ) { return putAt( indicesOfIndex(index), value ); }

    private void _setDataAt( int i, V o ) {
        if ( this.isVirtual() && i > 0 )
            throw new IllegalArgumentException("There is no data item at index "+i+" for this virtual tensor!");

        getDevice().access( this ).write( o ).at( i );
        _version++; // Autograd must be warned!
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setItems( Object value )
    {
        LogUtil.nullArgCheck( value, "value", Object.class );
        boolean success = true;
        if ( Number.class.isAssignableFrom(value.getClass()) ) { // A virtual tensor!
            this.setIsVirtual( true );
            value = DataConverter.get().convert( value, this.itemType() );
            this.getMut().setDataAt( 0, (V) value );
        }
        else if ( value.getClass().isArray() )
            getDevice().access(this).writeFrom( value );

        else success = false;

        if ( !success )
            _LOG.warn( "Failed to set value of type '"+value.getClass().getSimpleName()+"'!" );

        return this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Object getRawData() {
        _guardGet("data object");
        return _getData( true );
    }

    private Object _getData( boolean clone ) {
        Device<V> device = this.getDevice();
        if ( device == null ) return null;
        else return device.access( this ).readAll( clone );
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Object getRawItems() {
        _guardGet("value object");
        if ( this.getNDConf().isSimple() && !this.isSlice() )
            return getDevice().access(this).readAll(!this.isOutsourced());
        else
            return getDevice().access( this.deepCopy().setIsVirtual( false ) ).readAll(false);
    }

    /*==================================================================================================================
    |
    |       §(10) : Mapping :
    |   -----------------------------------------------------
    |       ...transformation and modification...
    */

    /**
     *  {@inheritDoc}
     */
    @Override
    public BufferedImage asImage( ImageType type )
    {
        switch ( type.bufferType )
        {
            case BufferedImage.TYPE_3BYTE_BGR: {
                _checkRankForImageConversion(type, Number.class, 0, 0, 3);
                // We expect a tensor of shape (height x width x 3)!
                BufferedImage image = new BufferedImage(shape(1), shape(0), type.bufferType);
                byte[] data = DataConverter.get().convert( _getRawData(), byte[].class);
                _writeImgData(new DataBufferByte(data, data.length), image);
                return image;
            }
            case BufferedImage.TYPE_4BYTE_ABGR:
            case BufferedImage.TYPE_4BYTE_ABGR_PRE:
            {
                _checkRankForImageConversion(type, Number.class, 0, 0, 4);
                BufferedImage image = new BufferedImage(shape(1), shape(0), type.bufferType);
                byte[] data = DataConverter.get().convert( _getRawData(), byte[].class);
                _writeImgData(new DataBufferByte(data, data.length), image);
                return image;
            }
            case BufferedImage.TYPE_INT_ARGB: {
                _checkRankForImageConversion(type, Number.class, 0, 0, 1);
                BufferedImage image = new BufferedImage(shape(1), shape(0), type.bufferType);
                int[] data = DataConverter.get().convert( _getRawData(), int[].class);
                _writeImgData(new DataBufferInt(data, data.length), image);
                return image;
            }
        }
        throw new IllegalArgumentException("Image type '"+type+"' not supported.");
    }

    private void _checkRankForImageConversion( ImageType type, Class<?> dataType, int... pattern ) {
        int rank = pattern.length; // The expected rank!
        if ( this.rank() != rank ) {
            throw new IllegalArgumentException(
                    "Cannot create image of type '" + type.name() + "' from tensor of rank " + this.rank() + ". " +
                    "Expected to receive tensor of rank " + rank + "."
                );
        }
        for ( int i = 0; i < pattern.length; i++ ) {
            int axisSize = pattern[ i ]; // The expected axis size!
            if ( axisSize > 0 ) {
                if ( axisSize != this.shape(i) ) {
                    String shape = this.shape().stream().map( a -> a.toString() ).collect(Collectors.joining("x"));
                    throw new IllegalArgumentException(
                        "Cannot create image of type '" + type.name() + "' from tensor with shape (" + shape + "). " +
                        "Axis " + i + " is expected to be of size " + axisSize + "."
                    );
                }
            }
        }
        if ( !dataType.isAssignableFrom(this.getItemType()) )
            throw new IllegalArgumentException(
                "Cannot create image of type '" + type.name() + "' from tensor of type '" + this.getItemType().getSimpleName() + ". " +
                "Expected to receive a tensor whose type is at least a sub-type of '" + dataType.getSimpleName() + "'."
            );
    }

    private static void _writeImgData( DataBuffer data, BufferedImage target ) {
        target.setData(
            Raster.createRaster( target.getSampleModel(), data, new Point() )
        );
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> addToGradient( Tsr<V> error ) {
        _guardSet("gradient");
        Optional<Tsr> grad = this.find( Tsr.class );
        grad.ifPresent( gradient ->
                            this.set(
                                    MemUtil.keep( gradient, error, () ->
                                            Neureka.get()
                                                    .backend()
                                                    .getFunction()
                                                    .plusAssign()
                                                    .call(gradient, error)
                                    )
                            ));
        if ( !grad.isPresent() ) {
            this.set( error );
            this.find( Device.class ).ifPresent( device -> {
                try {
                    device.store( error ) ;
                } catch ( Exception exception ) {
                    _LOG.error( "Failed trying to store a given error to a device for gradient accumulation.", exception );
                    throw exception;
                }
            });
        }
        return this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public <T> T asType( Class<T> typeClass )
    {
        LogUtil.nullArgCheck( typeClass, "typeClass", Class.class );
        if ( typeClass == Tsr.class ) return (T) this;
        if ( Number.class.isAssignableFrom( this.itemType()) && Number.class.isAssignableFrom(typeClass) ) {
            DataConverter converter = DataConverter.get();
            return converter.convert( mean().at(0).get(), typeClass );
        }
        if ( typeClass == String.class )
            return (T) this.toString();

        throw new IllegalArgumentException("Failed to convert this tensor of type '"+getDataType()+"' to '"+typeClass+"'!");
    }

    /**
     *  This method is an inline operation which changes the underlying data of this tensor.
     *  It converts the data types of the elements of this tensor to the specified type!<br>
     *  <br>
     *  <b>WARNING : The use of this method is discouraged for the following reasons: </b><br>
     *  <br>
     *  1. Inline operations are inherently error-prone for most use cases. <br>
     *  2. This inline operation in particular has no safety net,
     *     meaning that there is no implementation of version mismatch detection
     *     like there is for those operations present in the standard operation backend...
     *     No exceptions will be thrown during backpropagation! <br>
     *  3. This method has not yet been implemented to also handle instances which
     *     are slices of parent tensors!
     *     Therefore, there might be unexpected performance penalties or side effects
     *     associated with this method.<br>
     *     <br>
     *
     * @param typeClass The target type class for elements of this tensor.
     * @param <T> The type parameter for the returned tensor.
     * @return The same tensor instance whose data has been converted to hold a different type.
     */
    private <T> Tsr<T> _toType( Class<T> typeClass )
    {
        DataType<V> newDataType = (DataType<V>) DataType.of( typeClass );
        if ( newDataType != this.getDataType() ) {
            CPU.get().borrow((Tsr<Object>) this).in(()->{
                Object newData = _convertedDataOfType(typeClass);
                _setData( null );
                _setData( getDevice().allocateFromAll( newDataType, this.getNDConf(), newData) );
                return null;
            });
        }
        this.find( TsrImpl.class ).ifPresent( gradient -> gradient._toType( typeClass ) );
        return (Tsr<T>) this;
    }

    @Override
    public String toString()
    {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return NdaAsString.representing( this ).byDefaults().toString();
    }

    static int[][] makeFit( int[] sA, int[] sB ) {
        int lastIndexOfA = 0;
        for ( int i = sA.length-1; i >= 0; i-- ) {
            if ( sA[ i ] != 1 ) {
                lastIndexOfA = i;
                break;
            }
        }
        int firstIndexOfB = 0;
        for ( int i = 0; i < sB.length; i++ ) {
            if ( sB[ i ] != 1 ) {
                firstIndexOfB = i;
                break;
            }
        }
        int newSize = lastIndexOfA + sB.length - firstIndexOfB;
        int[] rsA = new int[ newSize ];
        int[] rsB = new int[ newSize ];
        for( int i = 0; i <newSize; i++ ) {
            if ( i <= lastIndexOfA ) rsA[ i ] = i; else rsA[ i ] = -1;
            if ( i >= lastIndexOfA ) rsB[ i ] = i - lastIndexOfA+firstIndexOfB; else rsB[ i ] = -1;
        }
        return new int[][]{ rsA, rsB };
    }

}
