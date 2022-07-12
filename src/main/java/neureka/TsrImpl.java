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
import neureka.backend.main.memory.MemUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.common.composition.AbstractComponentOwner;
import neureka.common.composition.Component;
import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.dtype.DataType;
import neureka.fluent.slicing.SliceBuilder;
import neureka.fluent.slicing.SmartSlicer;
import neureka.framing.NDFrame;
import neureka.framing.Relation;
import neureka.framing.fluent.AxisFrame;
import neureka.ndim.AbstractTensor;
import neureka.ndim.Filler;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterator.NDIterator;
import neureka.view.TsrAsString;
import org.jetbrains.annotations.NotNull;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.*;
import java.util.List;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;


/**
 *  The implementation for the {@link Tsr} API.
 *
 * @param <V> The type parameter for the individual value items within this tensor.
 */
final class TsrImpl<V> extends AbstractTensor<Tsr<V>, V>
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
    private static final byte IS_OUTSOURCED_MASK      = 2;
    private static final byte IS_VIRTUAL_MASK         = 4;
    private static final byte GRADIENT_APPLY_RQD_MASK = 8;
    private static final byte IS_DELETED_MASK         = 16;
    private static final byte IS_INTERMEDIATE_MASK    = 32;

    /**
     *  This integer represents the version of the data (accessible through {@link #getData()})
     *  stored within this tensor.
     *  It gets incremented every time an inline operation occurs!
     *  {@link GraphNode} instances tied to this tensor (as component) store
     *  a reference version which is a copy of this field.
     *  If this version changes, despite there being a GraphNode which might
     *  perform auto-differentiation at some point, then an exception will be thrown for debugging.
     *  <br>
     *  The corresponding getter returns the version of the data (accessible through {@link #getData()})
     *  stored within this tensor.
     */
    private int _version = 0;


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
            boolean success = t.createConstructionAPI().constructAllFromOne( NDConstructor.of(new int[]{ 1 }), args[ 0 ] );
            if ( !success ) {
                String message = "Cannot create tensor from argument of type '" + args[ 0 ].getClass().getName() + "'!";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
            return t;
        }
        args[ 0 ] = ( args[ 0 ] instanceof ArrayList ) ? ( (List<?>) args[ 0 ] ).toArray() : args[ 0 ];
        args[ 1 ] = ( args[ 1 ] instanceof ArrayList ) ? ( (List<?>) args[ 1 ] ).toArray() : args[ 1 ];
        if ( args[ 0 ] instanceof Object[] ) {
            if ( ( (Object[]) args[ 0 ] )[ 0 ] instanceof Integer || ((Object[])args[ 0 ])[ 0 ] instanceof Double) {
                args[ 0 ] = DataConverter.get().convert( args[ 0 ], int[].class );
            }
        }
        //CASES:
        if ( args[ 0 ] instanceof int[] ) {
            TsrImpl<T> t = new TsrImpl<>();
            if ( args[ 1 ] instanceof Double || args[ 1 ] instanceof Integer ) {
                args[ 1 ] = ( args[ 1 ] instanceof Integer ) ? ( (Integer) args[ 1 ] ).doubleValue() : args[ 1 ];
                t.createConstructionAPI().constructAllFromOne( NDConstructor.of((int[]) args[ 0 ]), args[ 1 ] );
            } else {
                t._setDataType( DataType.of( args[1].getClass() ) );
                t._constructAndAllocate( NDConstructor.of((int[]) args[0]), true );
                ((Object[])t.getUnsafe().getData())[0] = args[1];
            }
            return t;
        }
        /* EXPRESSION BASED CONSTRUCTION:
            The following allows the creation of tensors based on passing an expression
            alongside input tensors to the constructor.
            An example would be:

                Tsr<?> t = Tsr.of( "tanh(", x, ") * 7 ^", y );
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

    // Constructors:

    /**
     *  This constructor creates a completely empty tensor which is void of any contents and meaning.
     *  The use case for this would be to use the produced {@link Tsr}
     *  instance as a target for an inline operation which fills this instance with an actual value. <br>
     *  An example of this approach would be to call the {@link #putAt(List, Tsr)} method with an empty list as key.
     *  This will be interpreted as an inline copy of the contents of the
     *  second parameter into this {@link Tsr} instance.
     *  This constructor will be called by the {@link Tsr#newInstance()} factory method.
     */
    TsrImpl() {}

    TsrImpl( NDConstructor ndConstructor, DataType<?> dataType, Object data, boolean trusted ) {
        createConstructionAPI().tryConstructing( ndConstructor, dataType, data, trusted );
    }

    /**
     *  see {@link Tsr#of(DataType, int[], Filler)}
     */
    <T> TsrImpl( NDConstructor ndConstructor, DataType<T> type, Filler<T> filler ) {
        _constructFromInitializer(ndConstructor, type, filler);
    }

    /**
     *  See {@link Tsr#of(Class, int[], String)} and {@link #of(List, String)}
     */
    TsrImpl( Class<V> valueType, NDConstructor ndConstructor, String seed ) {
        createConstructionAPI().constructSeeded( valueType, ndConstructor, seed );
    }

    TsrImpl( NDConstructor ndConstructor, DataType<?> type ) {
        _setDataType( DataType.of( type.getRepresentativeType() ) );
        _constructAndAllocate(ndConstructor, true );
    }


    /**
     * @param ndConstructor The {@link NDConfiguration} producer of that this new tensor ought to have.
     * @param type The data type that this tensor ought to have.
     * @param filler The lambda Object which ought to fill this tensor with the appropriate data.
     * @param <T> The type parameter for the actual data array items.
     */
    private <T> void _constructFromInitializer(NDConstructor ndConstructor, DataType<T> type, Filler<T> filler ) {
        LogUtil.nullArgCheck(ndConstructor, "ndcProducer", NDConstructor.class );
        LogUtil.nullArgCheck( type, "type", DataType.class );
        LogUtil.nullArgCheck( type, "filler", Filler.class );
        _setDataType( type );
        _constructAndAllocate(ndConstructor, false );
        _initData(filler);
    }

    private void _constructAndAllocate(NDConstructor ndConstructor, boolean virtual ) {
        createConstructionAPI().configureFromNewShape(ndConstructor, virtual, true );
    }

    /*==================================================================================================================
    |
    |       §(2) : FLAGS
    |   ----------------------
    */

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setRqsGradient(boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient && !rqsGradient ) this.remove( TsrImpl.class );
        _setRqsGradient( rqsGradient );
        return this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public boolean rqsGradient() { return ( _flags & RQS_GRADIENT_MASK ) == RQS_GRADIENT_MASK; }

    private void _setRqsGradient(boolean rqsGradient) {
        if ( rqsGradient() != rqsGradient ) {
            if ( rqsGradient ) _flags += RQS_GRADIENT_MASK;
            else               _flags -= RQS_GRADIENT_MASK;
        }
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public boolean isIntermediate() { return ( _flags & IS_INTERMEDIATE_MASK ) == IS_INTERMEDIATE_MASK; }

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

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setIsOutsourced(boolean isOutsourced ) {
        _setIsOutsourced( isOutsourced );
        if ( isOutsourced )
            _setData( null );
        else if (
            !forComponent(
                Device.class,
                device -> {
                    try {
                        if ( device.has( this ) ) device.restore( this );
                    } catch ( Exception exception ) {
                        _LOG.error(
                            "Tensor could not be restored from device component when trying to migrate it back to RAM.",
                            exception
                        );
                        throw exception;
                    }
                    this.remove( Device.class );
                    forComponent(
                        TsrImpl.class,
                        gradient ->
                            ( (TsrImpl<V>) gradient ).forComponent(
                                Device.class,
                                gradDevice -> {
                                    try {
                                        if ( gradDevice.has( gradient ) ) gradDevice.restore( gradient );
                                    }
                                    catch ( Exception exception ) {
                                        _LOG.error(
                                            "Gradient could not be restored from device component when trying to migrate it back to RAM.",
                                            exception
                                        );
                                        throw exception;
                                    }
                                    gradient.remove( Device.class );
                                })
                    );
                }
            ) && _getData() == null
        ) {
            _setIsVirtual( true );
            _allocate( 1 ); // Only a single value representing the rest.
        }
        return this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public boolean isOutsourced() { return ( _flags & IS_OUTSOURCED_MASK ) == IS_OUTSOURCED_MASK; }

    private void _setIsOutsourced(boolean isOutsourced) {
        if ( isOutsourced() != isOutsourced ) {
            if ( isOutsourced ) _flags += IS_OUTSOURCED_MASK;
            else                _flags -= IS_OUTSOURCED_MASK;
        }
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public boolean isVirtual() { return ( _flags & IS_VIRTUAL_MASK ) == IS_VIRTUAL_MASK; }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> setIsVirtual( boolean isVirtual ) {

        assert getNDConf() != null;

        if ( isVirtual() != isVirtual ) {
            // Currently, we avoid offloading the virtualization by restoring outsourced tensors into RAM...
            Device<V> device = this.get( Device.class );
            try {
                // TODO: Fix this im-performant mess below:
                if ( device != null ) device.restore( this );
            } catch ( Exception exception ) {
                _LOG.error(
                    "Tensor could not be restored from device component when changing flag 'isVirtual' to " + isVirtual + ".",
                    exception
                );
                throw exception;
            }
            if ( isVirtual ) {
                if ( _getData() != null ) _virtualize();
            }
            else _actualize();
            // Virtual and actual tensors require a different mapping from a given index to the underlying data..
            // Therefore, we need to re-initialize the NDConfiguration object:
            createConstructionAPI().configureFromNewShape( NDConstructor.of(getNDConf().shape()), isVirtual, _getData() == null );
            if ( isVirtual ) {
                Relation<V> relation = get( Relation.class );
                if ( relation!=null )
                    relation.foreachChild( c -> {
                                ((TsrImpl)c)._setData( _getData());
                                c.setIsVirtual( true );
                            });
            } else {
                Tsr<?> parentTensor = ( this.isSlice() ) ? get(Relation.class).getParent() : null;
                if ( parentTensor != null ) parentTensor.get( Relation.class ).remove( this );
            }

            try {
                if ( device != null ) device.store( this );
            } catch ( Exception exception ) {
                String message =
                        "Tensor could not be migrated back to host device after changing flag 'isVirtual' to "+isVirtual+".";
                _LOG.error(
                        message,
                        exception
                );
                throw new IllegalStateException( message );
            }
        }
        else if ( isVirtual && _getData() == null ) _allocate( 1 ); //> Only a single value representing the rest.
        return this;
    }

    /**
     *  This method is the inner counterpart to the public "{@link Tsr#setIsVirtual}" method.
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

    /**
     *  {@inheritDoc}
     */
    @Override
    public boolean isDeleted() { return ( _flags & IS_DELETED_MASK ) == IS_DELETED_MASK; }

    /** {@inheritDoc} */
    @Override
    public boolean gradientApplyRequested() { return ( _flags & GRADIENT_APPLY_RQD_MASK ) == GRADIENT_APPLY_RQD_MASK; }

    /**
     *  {@inheritDoc}
     */
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
     *  In that case calling the "{@link Tsr.Unsafe#delete()}" method will free the memory reserved for this tensor.
     *  This manual memory freeing through this method can be faster than waiting for
     *  the garbage collector to kick in... <br>
     *  <br>
     *
     * @return This very tensor instance to allow for method chaining.
     */
    private Tsr<V> _delete()
    {
        if ( isDeleted() ) return this;
        forComponent( GraphNode.class, n -> {
            if ( n.isUsedAsDerivative() ) {
                String message = "Cannot delete a tensor which is used as derivative by the AD computation graph!";
                _LOG.error( message );
                throw new IllegalStateException( message );
            }
        });
        forComponent( Device.class, device -> device.free( this ) );
        _setData( null );
        _setNDConf( null );
        _flags = 0;
        forComponent( TsrImpl.class, t -> t.getUnsafe().delete() );
        _deleteComponents();
        _flags += IS_DELETED_MASK;

        return this;
    }

    /*==================================================================================================================
    |
    |       §(3) : COMPONENT SYSTEM
    |   --------------------------------
    */

    @Override
    public <T extends Component<?>> T get( Class<T> componentClass ) {
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

    private void _toLayout( NDConfiguration.Layout target )
    {
        if ( target == this.getNDConf().getLayout() ) return;

        NDConfiguration old = this.getNDConf();

        if ( target == NDConfiguration.Layout.ROW_MAJOR )
            _fromCMToRM();
        else
            _fromRMToCM();

        _checkLayoutConversion( this.getNDConf(), old, target );
    }

    /**
     *  Converts this tensor from column major to column major layout.
     */
    private void _fromCMToRM() {
        if ( this.getNDConf().isVirtual() ) {
            this.setIsVirtual( false ); // We actualized the tensor before conversion!
            if ( this.getNDConf().getLayout() == NDConfiguration.Layout.ROW_MAJOR )
                return;
        }
        TsrImpl<V> clone = deepCopy(); // A clone will have by default a row major layout.
        _setNDConf( clone.getNDConf() );
        _assignIfActual( () -> clone );
    }

    /**
     *  Converts this tensor from row major to column major layout.
     */
    private void _fromRMToCM() {
        _assignIfActual( () -> TsrImpl.this.T().deepCopy().getUnsafe().detach() );
        NDConfiguration old = this.getNDConf();
        int[] newTranslation = NDConfiguration.Layout.COLUMN_MAJOR.newTranslationFor(old.shape());
        if ( old.isVirtual() ) {
            this.setIsVirtual(false);
            old = this.getNDConf();
        }
        _setNDConf( _createNewNDCFrom( old, newTranslation, old.translation() ) );
    }

    /**
     *  This will only call the supplier and copy its result into this tensor
     *  if this tensor is not virtual (meaning this is an actual tensor).
     */
    private void _assignIfActual( Supplier<Tsr<?>> provider ) {
        if ( !this.isVirtual() ) {
            Tsr<?> toBeAssigned = provider.get();
            MemUtil.keep(this, toBeAssigned,
                () -> Neureka.get().backend().getFunction().idy().execute( this, toBeAssigned )
            );
        }
    }

    private static NDConfiguration _createNewNDCFrom(
        NDConfiguration old, int[] newTranslation, int[] indicesMap
    ) {
        assert !old.isVirtual();
        return NDConfiguration.of(
            old.shape(), newTranslation, indicesMap, old.spread(), old.offset()
        );
    }

    private static void _checkLayoutConversion(
            NDConfiguration newConf,
            NDConfiguration oldConf,
            NDConfiguration.Layout targetLayout
    ) {
        if ( newConf.isVirtual() )
            throw new IllegalStateException("Layout conversion produced a virtual nd-configuration!");
        if ( !newConf.getLayout().isCompatible(targetLayout) )
            throw new IllegalArgumentException(
                "Failed to convert this tensor from its original layout '"+oldConf.getLayout()+"' " +
                "to target layout '"+targetLayout+"'. Instead this tensor has layout '"+newConf.getLayout()+"'."
            );
    }

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
            _version++;
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
    private void _become(TsrImpl<V> tensor)
    {
        if ( tensor == null ) return;
        _setDataType( tensor.getDataType() );
        _setData( tensor.getUnsafe().getData() );
        _setNDConf( tensor.getNDConf() );
        _flags = tensor._flags;
        _transferFrom( tensor );
        tensor._setData( null );
        tensor._setDataType( null );
        tensor._setNDConf( null );
        tensor._flags = 0;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Unsafe<V> getUnsafe() {
        _guardGet("unsafe API");
        return new Unsafe<V>() {
            @Override
            public Tsr<V> setNDConf(NDConfiguration configuration ) { TsrImpl.this._setNDConf( configuration ); return TsrImpl.this; }
            @Override
            public <V> Tsr<V> toType( Class<V> typeClass ) { return TsrImpl.this._toType( typeClass ); }

            @Override
            public <U> Tsr<U> upcast( Class<U> superType ) {
                if ( superType.isAssignableFrom(TsrImpl.this.itemClass()) )
                    return (Tsr<U>) TsrImpl.this;
                else
                    throw new IllegalArgumentException("Provided type '"+superType+"' is not a super type of '"+ TsrImpl.this.itemClass()+"'.");
            }

            @Override
            public <T> Tsr<T> setDataType( DataType<T> dataType ) { return (TsrImpl<T>) TsrImpl.this._setDataType(dataType); }
            @Override
            public Tsr<V> toLayout(NDConfiguration.Layout layout) { TsrImpl.this._toLayout( layout ); return TsrImpl.this; }
            @Override
            public Tsr<V> incrementVersion(ExecutionCall<?> call ) {
                _incrementVersionBecauseOf( call );
                return TsrImpl.this;
            }
            @Override
            public Tsr<V> setIsIntermediate( boolean isIntermediate ) { return _setIsIntermediate( isIntermediate ); }
            @Override public Tsr<V> delete() { return TsrImpl.this._delete(); }
            @Override public Object getData() { return _getData(); }
            @Override
            public <A> A getDataAs( Class<A> arrayTypeClass ) {
                return DataConverter.get().convert( _getData(false), arrayTypeClass );
            }
            @Override
            public Tsr<V> setDataAt(int i, V o ) {
                _guardMod("data object");
                _setDataAt( i, o );
                return TsrImpl.this;
            }
            @Override public Tsr<V> detach() { TsrImpl.this.remove( GraphNode.class ); return TsrImpl.this; }
        };
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
    @NotNull
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
    public Tsr<V> to( Device<?> device ){ super._set( device ); return this; }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> label( String tensorName, String[][] labels )
    {
        LogUtil.nullArgCheck(labels, "labels", String[][].class, "Tensors cannot be labeled 'null'!");
        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) {
            frame = new NDFrame( this.rank(), tensorName );
            set(frame);
        }
        assert labels.length <= this.rank();
        for( int i = 0; i < labels.length; i++ ) {
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
    public Tsr<V> label(List<List<Object>> labels ) {
        LogUtil.nullArgCheck(labels, "labels", List.class, "Tensors cannot be labeled 'null'!");
        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) set( new NDFrame( labels, null ) );
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> label( String tensorName, List<List<Object>> labels ) {
        LogUtil.nullArgCheck(labels, "labels", List.class, "Tensors cannot be labeled 'null'!");
        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) set( new NDFrame<>( labels, tensorName ) );
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> label( Map<Object, List<Object>> labels )
    {
        LogUtil.nullArgCheck(labels, "labels", Map.class, "Tensors cannot be labeled 'null'!");
        this.set( new NDFrame<>( labels, this, null ) );
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Tsr<V> label(String tensorName, Map<Object, List<Object>> labels )
    {
        LogUtil.nullArgCheck(labels, "labels", Map.class, "Tensors cannot be labeled 'null'!");
        this.set( new NDFrame<>( labels, this, tensorName ) );
        return this;
    }

    /*==================================================================================================================
    |
    |       §(8) : (OVERLOADABLE) OPERATORS & OPERATIONS :
    |   -----------------------------------------------------
    |       ...for more context see package 'calculus'...
    |*/

    /** {@inheritDoc} */
    @Override
    public boolean isCase( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform 'is case' operation when second oprand is 'null'!");
        boolean[] found = { false };
        this.forComponent( Relation.class, r -> r.foreachChild( c -> {
                if ( c.equals( other ) ) found[ 0 ] = true;
            }));
        return found[ 0 ];
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
    public Tsr<V> getAt( Map<?,Integer> rankToStrides ) {
        LogUtil.nullArgCheck(rankToStrides, "rankToStrides", Map.class, "Rank-to-strides map must not be 'null'!");
        // ...not a simple slice... Advanced:
        return SmartSlicer.slice(
                        new Object[]{rankToStrides},
                        this,
                        this::_sliceOf
                    );
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
                this,
                this::_sliceOf
            );
    }

    /**
     *  If this tensor stores value types then this method will
     *  essentially produce a deep copy of this tensor.
     *  If the stored elements are reference types on the other hand,
     *  then the resulting clone may not be treated as a deep copy,
     *  especially if elements are mutable objects.
     *
     * @return A deep copy of this tensor.
     */
     @Override
    public TsrImpl<V> deepCopy() {
        Function cloner = Neureka.get().backend().getFunction().idy();
        boolean thisIsIntermediate = this.isIntermediate();
        _setIsIntermediate( false );
        Tsr<V> clone = Tsr.of( this.getItemClass() )
                            .on(this.getDevice())
                            .withShape( this.getNDConf().shape() )
                            .all( (V) Double.valueOf(0.0) );

        clone = cloner.call( clone, this );
        clone.getUnsafe().setIsIntermediate( thisIsIntermediate );
        _setIsIntermediate( thisIsIntermediate );
        return (TsrImpl<V>) clone;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public SliceBuilder<V> slice() { return new SliceBuilder<>( this, this::_sliceOf ); }

    /**
     *  This method is where the creation of a slice occurs.
     *  When creating a slice via the {@link SliceBuilder} or simply by passing ranges in the form of
     *  arrays, lists or maps to a {@link Tsr#getAt}(...) method, then this method will be called eventually.
     *  The creation of a slice always requires information about the shape of the new slice
     *  its position within the original tensor and also the strides / steps.
     *
     * @param newShape The of the slice which ought to be created.
     * @param newOffset The position of the new slice within this tensor.
     * @param newSpread The spread / steps / strides of the slice within this tensor.
     * @return The newly created slice.
     */
    private Tsr<V> _sliceOf( int[] newShape, int[] newOffset, int[] newSpread )
    {
        boolean isIntermediate = this.isIntermediate();
        _setIsIntermediate(false); // To avoid deletion!
        Tsr<V> slice = Function.of("slice(I[0])", false)
                        .with(Arg.Shape.of(newShape),Arg.Offset.of(newOffset),Arg.Stride.of(newSpread))
                        .call(this);

        slice.getUnsafe().setIsIntermediate(false);
        _setIsIntermediate(isIntermediate);
        return slice;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> putAt( List<?> key, Tsr<V> value ) {
        _putAtCheckFor( value );
        Tsr<V> slice = ( key == null ) ? this : getAt( key );
        return _putAt( slice, value );
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> putAt(int[] indices, V item ) {
        if ( indices == null )
            throw new IllegalArgumentException( "Provided indices are null!" );
        if ( indices.length > this.rank() ) {
            int[] correct = new int[rank()];
            System.arraycopy( indices, 0, correct, 0, indices.length );
            indices = correct;
        }
        if ( this.isVirtual() ) this.setIsVirtual( false );
        int i = getNDConf().indexOfIndices(indices);
        getUnsafe().setDataAt( i, item );
        return this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Tsr<V> putAt( Map<?,Integer> key, Tsr<V> value ) {
        _putAtCheckFor( value );
        Tsr<V> slice = ( key == null ) ? this : getAt( key );
        return _putAt( slice, value );
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

    private void _setDataAt( int i, V o ) {
        if ( this.isVirtual() && i > 0 )
            throw new IllegalArgumentException("There is no data item at index "+i+" for this virtual tensor!");

        getDevice().access( this ).write( o ).at( i );
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
            value = DataConverter.get().convert( value, this.itemClass() );
            this.getUnsafe().setDataAt( 0, (V) value );
        } else if ( value.getClass().isArray() ) {
            if ( this.isOutsourced() ) getDevice().access(this).writeFrom( value );
            else {
                if ( _getData() == null ) {
                    // This usually happens when a tensor was just freed from a device.
                    // We need to set its data again so that it is no longer empty...
                    _setData( value );
                    return this;
                } else {
                    getDevice().access(this).writeFrom(value);
                    setIsVirtual(false);
                }
            }
        }
        else success = false;

        if ( !success )
            _LOG.warn( "Failed to set value of type '"+value.getClass().getSimpleName()+"'!" );

        return this;
    }

    /**
     *  {@inheritDoc}
     */
    @Override
    public Object getData() {
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
    public Object getItems() {
        _guardGet("value object");
        if ( this.isOutsourced() ) {
            Device<V> device = get( Device.class );
            if ( device != null ) {
                if ( this.getNDConf().isSimple() && !this.isSlice() )
                    return device.access(this).readAll(false);
                else
                    return device.access( this.deepCopy().setIsVirtual( false ) ).readAll(false);
            }
        }
        if ( this.isVirtual() ) {
            if ( _getData() == null ) return null;
            else return getDevice().access(this).actualize();
        }
        else if ( this.getNDConf().isSimple() && !this.isSlice() )
            return getData();
        else
            return this.deepCopy().getUnsafe().getData();
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
                byte[] data = DataConverter.get().convert( _getData(), byte[].class);
                _writeImgData(new DataBufferByte(data, data.length), image);
                return image;
            }
            case BufferedImage.TYPE_4BYTE_ABGR:
            case BufferedImage.TYPE_4BYTE_ABGR_PRE:
            {
                _checkRankForImageConversion(type, Number.class, 0, 0, 4);
                BufferedImage image = new BufferedImage(shape(1), shape(0), type.bufferType);
                byte[] data = DataConverter.get().convert( _getData(), byte[].class);
                _writeImgData(new DataBufferByte(data, data.length), image);
                return image;
            }
            case BufferedImage.TYPE_INT_ARGB: {
                _checkRankForImageConversion(type, Number.class, 0, 0, 1);
                BufferedImage image = new BufferedImage(shape(1), shape(0), type.bufferType);
                int[] data = DataConverter.get().convert( _getData(), int[].class);
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
        if ( !dataType.isAssignableFrom(this.getItemClass()) )
            throw new IllegalArgumentException(
                "Cannot create image of type '" + type.name() + "' from tensor of type '" + this.getItemClass().getSimpleName() + ". " +
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
        if (
            !forComponent(
                Tsr.class,
                    gradient ->
                    this.set(
                        MemUtil.keep( gradient, error, () ->
                            Neureka.get()
                                    .backend()
                                    .getFunction()
                                    .plusAssign()
                                    .call(gradient, error)
                        )
                    )
            )
        ) {
            this.set( error );
            this.forComponent( Device.class, device -> {
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
        if ( Number.class.isAssignableFrom( this.itemClass()) && Number.class.isAssignableFrom(typeClass) ) {
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
        if ( this.isOutsourced() ) {
            _setDataType( DataType.of( typeClass ) );
            return (Tsr<T>) this;
        }
        else {
            Object newData = _convertedDataOfType( typeClass );
            _setData( null );
            _setDataType( DataType.of( typeClass ) );
            _setData( newData );
        }
        forComponent( TsrImpl.class, gradient -> gradient._toType( typeClass ) );
        return (Tsr<T>) this;
    }

    @Override
    public String toString()
    {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return TsrAsString.representing( this ).byDefaults().toString();
    }

}
