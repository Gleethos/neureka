package neureka;

import groovy.lang.IntRange;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.other.Reshape;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.*;
import neureka.ndim.AbstractNDArray;
import neureka.devices.host.HostCPU;
import neureka.devices.Device;
import neureka.framing.IndexAlias;
import neureka.framing.Relation;
import neureka.calculus.Function;
import neureka.calculus.frontend.assembly.FunctionBuilder;
import neureka.autograd.GraphNode;
import neureka.autograd.JITProp;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterators.NDIterator;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;
import neureka.optimization.Optimizer;
import neureka.utility.DataConverter;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

public class Tsr<ValueType> extends AbstractNDArray<Tsr<ValueType>, ValueType> implements Component<Tsr<ValueType>>
{
    static {
        _CPU = HostCPU.instance();
        _LOGGER = LoggerFactory.getLogger( Tsr.class );
    }


    private static Logger _LOGGER; // Why is this not final ? : For unit testing!

    /**
     *  Default device (host cpu)
     */
    private static final Device<Number> _CPU;

    /**
     *  Flag Fields
     */
    private int _flags = 0;

    /**
     * This is a bit mask used to store true / false values
     * in a targeted bit inside the "_flags" variable.
     */
    private static final int RQS_GRADIENT_MASK = 1;
    private static final int IS_OUTSOURCED_MASK = 2;
    private static final int IS_VIRTUAL_MASK = 4;
    private static final int GRADIENT_APPLY_RQD_MASK = 8;

    /**
     *  The version of the data (_value) stored within this tensor.
     *  Gets incremented every time an inline operation occurs!
     */
    private int _version = 0;

    /**
     *  This method returns the version of the data (_value) stored within this tensor.
     *
     * @return The version of the underlying data of this tensor.
     */
    public int version() {
        return _version;
    }

    /**
     *  This method is responsible for incrementing
     *  the "_version" field variable which represents the version of the data of this tensor.
     *  Meaning :
     *  Every time the underlying data (_value) changes this version ought to increment alongside.
     *  The method is called during the execution procedure.
     *
     * @param call The context object containing all relevent informatin that defines a call for tensor execution.
     * @return This very tensor instance. (factory pattern)
     */
    public Tsr<ValueType> incrementVersionBecauseOf( ExecutionCall call ){
        if ( Neureka.instance().settings().autograd().isPreventingInlineOperations() ) {
            _version ++;
            GraphNode node = find( GraphNode.class );
            if ( node != null && node.referenceVersion() != this._version ) {
                if ( node.usesAD() || node.isUsedAsDerivative() ) {
                    String error = "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\n" +
                            "The following OperationType caused an internal version mismatch: '"+call.getType().getFunction()+"'";
                    _LOGGER.error( error );
                    throw new IllegalStateException( error );
                }
            }
        }
        return this;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public Tsr<ValueType> setRqsGradient( boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient && !rqsGradient ) this.remove( Tsr.class );
        _setRqsGradient( rqsGradient );
        return this;
    }

    public boolean rqsGradient() {
        return ( _flags & RQS_GRADIENT_MASK ) == RQS_GRADIENT_MASK;
    }

    protected void _setRqsGradient( boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient ) {
            if ( rqsGradient ) _flags += RQS_GRADIENT_MASK;
            else _flags -= RQS_GRADIENT_MASK;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    public Tsr<ValueType> setIsOutsourced( boolean isOutsourced ) {
        _setIsOutsourced( isOutsourced );
        if ( isOutsourced ) {
            _value = null;
        } else if (
                !forComponent(
                    Device.class,
                    d -> {
                        try {
                            if ( d.has( this ) ) d.restore( this );
                        } catch ( Exception exception ) {
                            _LOGGER.error(
                                    "Tensor could not be restored from device component when trying to migrate it back to RAM.",
                                    exception
                            );
                            throw exception;
                        }
                        this.remove( Device.class );
                        forComponent(
                            Tsr.class,
                            gradient ->
                            gradient.forComponent(
                                Device.class,
                                gd -> {
                                    try {
                                        if ( ( (Device) gd ).has( gradient ) ) ( (Device) gd ).restore( gradient );
                                    } catch ( Exception exception ) {
                                        _LOGGER.error(
                                                "Gradient could not be restored from device component when trying to migrate it back to RAM.",
                                                exception
                                        );
                                        throw exception;
                                    }
                                    gradient.remove( Device.class );
                                })
                        );
                    }
                ) && _value == null
        ){
            setIsVirtual( true );
        }
        return this;
    }

    public boolean isOutsourced() {
        return ( _flags & IS_OUTSOURCED_MASK ) == IS_OUTSOURCED_MASK;
    }

    protected void _setIsOutsourced( boolean isOutsourced ) {
        if ( isOutsourced() != isOutsourced ) {
            if ( isOutsourced ) _flags += IS_OUTSOURCED_MASK;
            else _flags -= IS_OUTSOURCED_MASK;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    public Tsr<ValueType> setIsVirtual( boolean isVirtual ) {
        if ( isVirtual() != isVirtual ) {
            Device device = this.find( Device.class );
            try {
                if ( device != null ) device.restore( this );
            } catch ( Exception exception ) {
                _LOGGER.error(
                        "Tensor could not be restored from device component when changing flag 'isVirtual' to " + isVirtual + "."
                        , exception
                );
                throw exception;
            }
            double v = ( _value == null ) ? 0 : ((this.is64())?((double[])_value)[ 0 ]:((float[])_value)[ 0 ]);
            if ( isVirtual ) {
                _value = new double[]{v};
                Relation<ValueType> relation = find( Relation.class );
                if ( relation!=null ) relation.foreachChild( c -> c._value=_value);
            } else {
                Tsr<?> parentTensor = (this.isSlice())? find(Relation.class).getParent() : null;
                if ( parentTensor != null ) {
                    parentTensor.find( Relation.class ).remove( this );
                }
                _value = (this.is64()) ? new double[ this.size() ] : new float[ this.size() ];
                int length = (this.is64()) ? ((double[]) _value).length : ((float[]) _value).length;
                for (int i = 0; i < length; i++) {
                    if (this.is64()) ((double[]) _value)[i] = v;
                    else ((float[]) _value)[i] = (float) v;
                }
            }
            _setIsVirtual( isVirtual );
            if( _conf != null ) _configureFromNewShape( _conf.shape(), isVirtual );
            try {
                if( device != null ) device.store( this );
            } catch ( Exception exception ) {
                String message =
                        "Tensor could not be migrated back to host device after changing flag 'isVirtual' to "+isVirtual+".";
                _LOGGER.error(
                        message,
                        exception
                );
                throw new IllegalStateException( message );
            }
        } else if ( isVirtual && _value == null ) _value = new double[]{0};
        return this;
    }

    public boolean isVirtual() {
        return (_flags & IS_VIRTUAL_MASK) == IS_VIRTUAL_MASK;
    }

    /**
     *  This method is the inner counterpart to the public "setIsVirtual" method.
     *  It actually performs the bit flipping by applying the corresponding bit mask.
     * @param isVirtual The truth value which ought to be applied.
     */
    protected void _setIsVirtual( boolean isVirtual ) {
        if ( isVirtual() != isVirtual ) {
            if ( isVirtual ) _flags += IS_VIRTUAL_MASK;
            else _flags -= IS_VIRTUAL_MASK;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    public Tsr<ValueType> setGradientApplyRqd( boolean applyRequested ) {
        if ( gradientApplyRqd() != applyRequested ) {
            if ( applyRequested ) {
                if (
                        Neureka.instance().settings().autograd().isApplyingGradientWhenRequested() &&
                        !Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed()
                ) {
                    this.applyGradient();
                } else _flags += GRADIENT_APPLY_RQD_MASK;
            }
            else _flags -= GRADIENT_APPLY_RQD_MASK;
        }
        return this;
    }

    public boolean gradientApplyRqd() {
        return (_flags & GRADIENT_APPLY_RQD_MASK) == GRADIENT_APPLY_RQD_MASK;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     * This method is executed when a new Component is added to the tensor.
     * The public add method is implemented in the super class
     * 'AbstractComponentOwner' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. (GraphNode, IndexAlias, Relation, int[], ...)
     * @return The unchanged object or maybe in future versions: null (component rejected)
     */
    @Override
    protected < T extends Component<Tsr<ValueType>> > T _addOrReject( T newComponent )
    {
        if ( newComponent.getClass() == HostCPU.class ) return null;
        if ( newComponent instanceof Device && !( (Device) newComponent ).has( this ) )
        {
            if ( this.has( Relation.class ) ) {
                Relation relation = find( Relation.class );
                if ( relation.hasParent() ) { // Root needs to be found ! :
                    Tsr<ValueType> root = relation.findRootTensor();
                    try {
                        ((Device)newComponent).store( root );
                    } catch ( Exception exception ) {
                        _LOGGER.error( "Could not store tensor on device '" + newComponent.toString() +"'.", exception );
                        throw exception;
                    }
                    root.find( Relation.class ).foreachChild( c -> ((Tsr)c).setIsOutsourced( true ) );
                } else { // This is root ! :
                    relation.foreachChild( c -> ((Tsr<?>)c).setIsOutsourced( true ) );
                    try {
                        ((Device)newComponent).store( this );
                    } catch ( Exception exception ) {
                        _LOGGER.error( "Could not store tensor on device '" + newComponent.toString() +"'.", exception );
                        throw exception;
                    }
                }
            } else {
                try {
                    ((Device)newComponent).store( this );
                } catch ( Exception exception ) {
                    _LOGGER.error( "Could not store tensor on device '" + newComponent.toString() +"'.", exception );
                    throw exception;
                }
            }
            if ( ((Device)newComponent).has( this ) ) setIsOutsourced( true );
        } else if ( newComponent instanceof Tsr ) {
            if (
                    ((Tsr)newComponent).shape().hashCode() != this.shape().hashCode() ||
                    Arrays.hashCode(((Tsr)newComponent).getNDConf().shape()) != Arrays.hashCode( _conf.shape() )
            ) newComponent = null;
        }
        return newComponent;
    }

    /**
     * This method is executed when a component is being removed from the tensor.
     * The public remove method is implemented in the super class
     * 'AbstractComponentOwner' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. (GraphNode, IndexAlias, Relation, int[], ...)
     * @return The unchanged object or when rejected: null (component rejected)
     */
    @Override
    protected <T extends Component<Tsr<ValueType>>> T _removeOrReject( T newComponent )
    {
        if ( newComponent instanceof Device ) {

        }
        return newComponent;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // HIGH LEVEL PROPERTIES :

    public boolean isEmpty() {
        return _value == null && !this.isOutsourced();
    }

    public boolean isUndefined() {
        return _conf==null || _conf.shape() == null;
    }

    public boolean isSlice() {
        Relation<ValueType> child = find( Relation.class );
        return ( child != null && child.hasParent() );
    }

    public int sliceCount() {
        Relation<ValueType> child = find( Relation.class );
        return ( child != null ) ? child.childCount() : 0;
    }

    public boolean isSliceParent(){
        Relation<ValueType> parent = find( Relation.class );
        return ( parent != null && parent.hasChildren() );
    }

    public boolean belongsToGraph() {
        return this.has( GraphNode.class );
    }

    public boolean isLeave() {
        return (!this.has( GraphNode.class )) || this.find( GraphNode.class ).isLeave();
    }

    public boolean isBranch() {
        return !this.isLeave();
    }

    public boolean hasGradient() {
        return this.has( Tsr.class );
    }

    public Tsr<ValueType> getGradient() {
        return this.find( Tsr.class );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Direct Access to component (Device)

    /**
     * @return The device on which this tensor is stored or 'CPU' if it is not outsourced.
     */
    public Device<ValueType> device() {
        if ( this.isOutsourced() ) return this.find( Device.class );
        return (Device<ValueType>) _CPU;
    }

    /**
     *
     * @return The graph node of the computation graph to which this tensor belongs or null if not part of a graph.
     */
    public GraphNode<ValueType> getGraphNode(){
        return find( GraphNode.class );
    }

    /**
     *
     * @return Custom IndexAlias object.
     */
    public IndexAlias<ValueType> index(){
        return find( IndexAlias.class );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    protected Tsr<ValueType> _become( Tsr<ValueType> tensor ) {
        if ( tensor == null ) return this;
        _value = tensor._value;
        _type = tensor._type;
        _conf = tensor._conf;
        _components = Collections.synchronizedList( new ArrayList<>() );
        _flags = tensor._flags;
        if ( tensor._components != null ) { // Inform components about their new owner:
            _components.addAll( tensor._components );
            List<Component<Tsr<ValueType>>> snapshot = new ArrayList<>( tensor._components );
            for ( Component<Tsr<ValueType>> o : snapshot ) o.update( tensor, this );
        }
        tensor._value = null;
        tensor._type = null;
        tensor._conf = null;
        tensor._components = null;
        tensor._flags = -1;
        return this;
    }

    public Tsr<ValueType> delete() {
        forComponent( GraphNode.class, n -> {
            if ( n.isUsedAsDerivative() ) {
                String message = "Cannot delete a tensor which is used as derivative by the AD computation graph!";
                _LOGGER.error( message );
                throw new IllegalStateException( message );
            }
        });
        forComponent( Device.class, d -> d.free( this ) );
        _flags = -1;
        _value = null;
        _conf = null;
        forComponent( Tsr.class, Tsr::delete );
        _components = null;
        return this;
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    private int _dataLength(){
        if ( !(_value instanceof float[]) && !(_value instanceof double[]) ) {
            if ( _value instanceof Object[] ) return ((Object[])_value).length;
            else return -1;
        } else if ( this.is64() ) return ( (double[]) _value ).length;
        else return ( (float[]) _value ).length;
    }

    /**
     * @param newShape
     */
    protected void _configureFromNewShape( int[] newShape, boolean makeVirtual ) {
        int size = NDConfiguration.Utility.szeOfShp( newShape );
        _value = ( _value == null ) ? new double[ size ] : _value;
        int length = _dataLength();
        if ( length >= 0 ) {
            if ( size != length && ( !this.isVirtual() || !makeVirtual) ) {
                String message = "Size of shape does not match stored value64!";
                _LOGGER.error( message );
                throw new IllegalArgumentException( message );
            }
        }
        if ( makeVirtual ) {
            _conf = VirtualNDConfiguration.construct( newShape );
        } else {
            int[] newTranslation = NDConfiguration.Utility.newTlnOf( newShape );
            int[] newIdxmap = newTranslation;
            int[] newSpread = new int[ newShape.length ];
            Arrays.fill( newSpread, 1 );
            int[] newOffset = new int[ newShape.length ];
            _conf = AbstractNDC.construct( newShape, newTranslation, newIdxmap, newSpread, newOffset );
        }
    }


    //CONSTRUCTION :
    //=========================

    public Tsr(){}

    //Generic construction: ( Groovy, Jython, ... )
    public Tsr( Object arg ){
        _construct( new Object[]{ arg } );
    }

    public Tsr( String equation, List<Object> inputs ) {
        _construct(
                inputs.stream().map( Tsr::new ).toArray( Tsr[]::new ),
                equation,
                true
        );
    }

    public Tsr( List<?> arg1, String arg2 )
    {
        java.util.function.Function<Class, Boolean> isType = c -> arg1.stream().allMatch( e -> e.getClass() == c );

        if ( isType.apply( Integer.class ) ) {
            List<Integer> shape = (List<Integer>) arg1;
            int[] shp = new int[ shape.size() ];
            for ( int i=0; i < shp.length; i++ ) shp[ i ] = shape.get( i );
            _construct( shp, arg2 );
        } else if ( isType.apply( Tsr.class ) ) {
            _construct( arg1.toArray( new Tsr[ 0 ] ), arg2, true );
        } else {
            _construct(
                    ( (List<Object>) arg1 ).stream().map( Tsr::new ).toArray( Tsr[]::new ),
                    arg2,
                    true
            );
        }
    }

    public Tsr( List<Integer> shape, List<ValueType> range )
    {
        int[] shp = new int[ shape.size() ];
        for( int i=0; i<shp.length; i++ ) shp[ i ] = shape.get( i );
        if ( range.size() == 1 && range.get( 0 ) instanceof IntRange ) range = (List<ValueType>) range.get( 0 );

        if ( !range.isEmpty() && !( range.get( 0 ) instanceof Number ) ) {
            Class<?> givenClass = range.get( 0 ).getClass();
            @SuppressWarnings("unchecked")
            final ValueType[] value = (ValueType[]) Array.newInstance(
                    givenClass,
                    NDConfiguration.Utility.szeOfShp( shp )
            );
            for ( int i = 0; i < value.length; i++ ) value[ i ] = range.get( i % range.size() );
            _value = value;
            _type = DataType.instance( givenClass );
            _construct( shp, value );
        } else {
            double[] value = new double[ NDConfiguration.Utility.szeOfShp( shp ) ];
            _type = DataType.instance( F64.class );
            for ( int i = 0; i < value.length; i++ ) {
                if ( range.get( i % range.size() ) instanceof BigDecimal ) {
                    value[ i ] = ( (BigDecimal) range.get( i % range.size() ) ).doubleValue();
                } else if ( range.get( i % range.size() ) instanceof Integer ) {
                    value[ i ] = (Integer) range.get( i % range.size() );
                }
            }
            _construct( shp, value );
        }
    }

    private void _construct( int[] shape, ValueType[] value ) {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        if ( size != value.length ) {
            Class<?> givenClass = value[ 0 ].getClass();
            @SuppressWarnings("unchecked")
            final ValueType[] newValue = (ValueType[]) Array.newInstance(
                    givenClass,
                    NDConfiguration.Utility.szeOfShp( shape )
            );
            for ( int i = 0; i < newValue.length; i++ ) newValue[ i ] = value[ i % value.length ];
            _value = newValue;
            _type = DataType.instance( givenClass );
        }
        else _value = value;
        _configureFromNewShape( shape, false );
    }

    private void _construct( List<List<Object>> matrix ) {
        boolean isNumeric = matrix.stream().allMatch( e -> e.stream().allMatch( ie -> ie instanceof Number ) );
        if ( isNumeric ) {
            int n = matrix.get( 0 ).size();
            boolean isHomogenius = matrix.stream().allMatch( e -> e.size() == n );
            if ( isHomogenius ) {
                int m = matrix.size();
                double[] value = new double[ m * n ];
                boolean isLegacy = Neureka.instance().settings().indexing().isUsingLegacyIndexing();
                int[] shape = ( isLegacy )
                        ? new int[]{ n, m }
                        : new int[]{ m, n };

                for ( int mi = 0; mi < m; mi++ ) {
                    for ( int ni = 0; ni < n; ni++ ) {
                        int i = ( isLegacy ) ? m * ni + mi : n * mi + ni;
                        value[ i ] = DataConverter.instance().convert( matrix.get( mi ).get( ni ), Double.class );
                    }
                }
                _construct( shape, value );
            } else {
                String message = "Provided nested list(s) do not form a regular matrix.";
                _LOGGER.error( message );
                throw new IllegalArgumentException( message );
            }
        }
    }

    public Tsr( List<Object> conf ) {
        boolean isMatric = conf.stream().allMatch( e -> e instanceof List );
        if ( isMatric ) {
            _construct( conf.stream().map( e -> (List<Object>) e ).collect( Collectors.toList() ) );
            return;
        }
        boolean isNatural = ( conf.size() <= 64 );
        for( Object e : conf ) {
            if( !isNatural ) break;
            double asNum = ( e instanceof BigDecimal ) ?
                    ( (BigDecimal) e ).doubleValue()
                    : ( e instanceof Double )
                        ? (Double) e
                        : (Integer) e;
            isNatural = asNum % 1 == 0;
        }
        if ( isNatural ) {
            int[] shape = new int[ conf.size() ];
            for ( int i = 0; i < shape.length; i++ ) {
                shape[ i ] = ( conf.get( i ) instanceof BigDecimal )
                        ? ( (BigDecimal) conf.get( i ) ).intValue() :
                            ( conf.get( i ) instanceof Double )
                                    ? ( (Double) conf.get( i ) ).intValue()
                                    :( (Integer) conf.get( i ) );
            }
            _construct( shape );
        } else {
            double[] value = new double[ conf.size() ];
            for (int i = 0; i < value.length; i++ ) {
                value[ i ] = ( conf.get( i ) instanceof BigDecimal )
                        ? ( (BigDecimal) conf.get( i ) ).doubleValue() :
                            ( conf.get( i ) instanceof Double )
                                ? ( (Double) conf.get( i ) ).doubleValue()
                                : ( (Integer) conf.get( i ) );
            }
            _construct( new int[]{ conf.size() }, value );
        }

    }

    public Tsr( List<Integer> arg1, Object arg2 ) {
        _construct( new Object[]{ arg1, arg2 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3 ){
        _construct( new Object[]{ arg1, arg2, arg3 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, String arg4 ){
        _construct( new Object[]{ arg1, arg2, arg3, arg4 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5, arg6 } );
    }

    public Tsr ( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7 ) {
        _construct( new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7} );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8, Object arg9 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8, Object arg9, Object arg10 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10 } );
    }

    public Tsr( int[] shape, String seed ){
        _construct( shape, seed );
    }

    private void _construct( int[] shape, String seed ) {
        _construct( shape );
        _value = DataConverter.Utility.seededDoubleArray( (double[]) _value, seed );
    }

    private int[] _intArray( Object[] arg ) {
        int length = arg.length;
        int[] array = new int[ length ];
        for ( int i = 0; i < length; i++ ) {
            if ( arg[ i ] instanceof Double ) array[ i ] = ( (Double) arg[ i ] ).intValue();
            else array[ i ] = (Integer) arg[ i ];
        }
        return array;
    }

    private double[] _doubleArray( Object[] arg ) {
        int length = arg.length;
        double[] array = new double[ length ];
        for ( int i = 0; i < length; i++ ) {
            if ( arg[ i ] instanceof Integer ) array[ i ] = (Integer) arg[ i ];
            else if ( arg[ i ] instanceof Double ) array[ i ] = (Double) arg[ i ];
            else if ( arg[ i ] instanceof BigDecimal ) array[ i ] = ( (BigDecimal) arg[ i ] ).doubleValue();
        }
        return array;
    }

    public Tsr( Object[] args ){
        _construct( args );
    }

    private void _construct( Object[] args ) {
        if ( args == null || args.length == 0 ) return;
        if ( args.length == 1 ) {
            if ( args[ 0 ] instanceof Object[] ) {
                _construct( (Object[]) args[ 0 ] );
                return;
            } else if ( args[ 0 ] instanceof BigDecimal ) {
                _construct( new int[]{ 1 }, ( (BigDecimal) args[ 0 ] ).doubleValue());
                return;
            } else if ( args[ 0 ] instanceof Integer ) {
                _construct( new int[]{ 1 }, ( (Integer) args[ 0 ] ).doubleValue() );
                return;
            } else {
                String message = "Cannot create tensor from argument of type '" + args[ 0 ].getClass().getName() + "'!";
                _LOGGER.error( message );
                throw new IllegalArgumentException( message );
            }
        }
        args[ 0 ] = ( args[ 0 ] instanceof ArrayList ) ? ( (ArrayList) args[ 0 ] ).toArray() : args[ 0 ];
        args[ 1 ] = ( args[ 1 ] instanceof ArrayList ) ? ( (ArrayList) args[ 1 ] ).toArray() : args[ 1 ];
        if ( args[ 0 ] instanceof Object[] ) {
            if ( ( (Object[]) args[ 0 ] )[ 0 ] instanceof Integer || ((Object[])args[ 0 ])[ 0 ] instanceof Double) {
                args[ 0 ] = _intArray( (Object[]) args[ 0 ] );
            }
        }
        if ( args[ 1 ] instanceof Object[] ) {
            if ( ((Object[]) args[ 1 ] )[ 0 ] instanceof Integer ) args[ 1 ] = _doubleArray( (Object[]) args[ 1 ] );
            else if ( ( ( Object[] ) args[ 1 ] )[ 0 ] instanceof BigDecimal ) args[ 1 ] = _doubleArray( (Object[]) args[ 1 ] );
        }
        //CASES:
        if ( args[ 0 ] instanceof int[] && ( args[ 1 ] instanceof Double || args[ 1 ] instanceof Integer ) ) {
            args[ 1 ] = ( args[ 1 ] instanceof Integer ) ? ( (Integer) args[ 1 ] ).doubleValue() : args[ 1 ];
            _construct( (int[]) args[ 0 ], (Double) args[ 1 ] );
            return;
        } else if ( args[ 0 ] instanceof int[] && args[ 1 ] instanceof double[] ) {
            _construct( (int[]) args[ 0 ], (double[]) args[ 1 ] );
            return;
        }
        // EXPRESSION:
        boolean containsString = false;
        int numberOfTensors = 0;
        ArrayList<Tsr<ValueType>> tsrList = new ArrayList<>();
        for ( Object o : args ) {
            containsString = ( o instanceof String ) || containsString;
            if ( o instanceof Tsr ) {
                tsrList.add( (Tsr<ValueType>) o );
                numberOfTensors++;
            }
        }
        boolean doAD = true;
        Tsr<?>[] tsrs = new Tsr[ numberOfTensors ];
        StringBuilder f = new StringBuilder();
        int ti = 0;
        for ( Object o : args ) {
            if ( tsrList.contains( o ) ){
                tsrs[ ti ] = ( (Tsr<?>) o );
                f.append( "I[" ).append( ti ).append( "]" );
                ti++;
            } else if ( o instanceof  String ) f.append( (String) o );
            else if ( o instanceof  Boolean ) doAD = (Boolean) o;
        }
        _construct( tsrs, f.toString(), doAD );
    }

    public Tsr( double value ){
        _construct( new int[]{ 1 }, value );
    }

    public Tsr( int[] shape ) {
        _construct( shape );
    }

    private void _construct( int[] shape ) {
        _value = new double[ NDConfiguration.Utility.szeOfShp( shape ) ];
        _configureFromNewShape( shape, false );
    }

    public Tsr( int[] shape, double value ) {
        _construct( shape, value );
    }

    private void _construct( int[] shape, double value ) {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        _value = new double[ 1 ];
        _type = DataType.instance( F64.class );
        setIsVirtual( size > 1 );
        _configureFromNewShape( shape, size > 1 );
        ( (double[])_value )[ 0 ] = value;
    }

    public Tsr( int[] shape, double[] value ) {
         _construct( shape, value );
    }

    private void _construct( int[] shape, double[] value ) {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        if ( size != value.length ) {
            double[] newValue = new double[ size ];
            for ( int i = 0; i < newValue.length; i++ ) newValue[ i ] = value[ i % value.length ];
            _value = newValue;
        } else _value = value;
        _configureFromNewShape( shape, false );
    }

    public Tsr( int[] shape, DataType dataType, Object value ) {
        _value = value;
        _type = dataType;
        _configureFromNewShape( shape, false );
    }

    // TRACKED COMPUTATION :
    //=========================

    /**
     *  This method takes a tensor and a String expression describing
     *  operations which ought to be applied to said tensor.
     *  This expression will be parsed to a Function instance expecting one input,
     *  namely : "I[0]"
     *  An example would be the following :
     *  'Tsr a = new Tsr( b, "sin( I[0] ) * 2" )'
     *  Which takes the tensor 'b' and applies the function "f(x) = sin(x) * 2"
     *  elementwise to produce a new tensor 'a'!
     *
     * @param tensor A tensor which serves as input to the Function instance parsed from the given expression.
     * @param expression The expression describing operations applied to the provided tensor.
     */
    public Tsr( Tsr<ValueType> tensor, String expression ) {
        if ( tensor == null ) return;
        _construct( new Tsr[]{ tensor }, expression, true );
    }

    /**
     *  This method takes an array of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said array.
     *  This expression will be parsed to a Function instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ...
     *  An example would be the following :
     *  'Tsr a = new Tsr( new Tsr[]{ b, c }, "sin( I[0] ) / I[1]" )'
     *  Which takes the tensor 'b' & 'c' and applies the function "f(x,y) = sin(x) / y"
     *  elementwise to produce a new tensor 'a'!
     *
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     * @param expression The expression describing operations applied to the provided tensors.
     */
    public Tsr( Tsr<ValueType>[] tensors, String expression ) {
        _construct( tensors, expression, true );
    }

    /**
     *  This method takes an array of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said array.
     *  This expression will be parsed to a Function instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ...
     *  An example would be the following :
     *  'Tsr a = new Tsr( new Tsr[]{ b, c }, "sin( I[0] ) / I[1]" )'
     *  Which takes the tensor 'b' & 'c' and applies the function "f(x,y) = sin(x) / y"
     *  elementwise to produce a new tensor 'a'!
     *  Additionally there is a helpful flag which allows one to specify if the
     *  parsed Function instance emerging from the provided expression
     *  should also allow the tracking of computations via a computation graph (GraphNode instances).
     *  This history tracking then enables auto-differentiation.
     *
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     * @param expression The expression describing operations applied to the provided tensors.
     * @param doAD A flag which when set to true commands the creation of a computation graph during operation execution.
     */
    public Tsr( Tsr<ValueType>[] tensors, String expression, boolean doAD ) {
        _construct( tensors, expression, doAD );
    }

    private void _construct( Tsr[] tensors, String operation, boolean doAD ) {
        if ( tensors == null || tensors.length == 0 || tensors[ 0 ] == null ) return;
        Tsr<ValueType> result = Function.Setup.commit( this, tensors, operation, doAD );
        this._become( result );
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  This method performs various operations by calling Function instances
     *  in order to ultimately calculate the mean value of all values
     *  of this very tensor!
     *  This scalar tensor is then returned.
     *
     * @return A scalar tensor which is the mean value of all values of this very tensor.
     */
    public Tsr<ValueType> mean() {
        Tsr<ValueType> ones = new Tsr<>( this.getNDConf().shape(), 1 );
        Tsr<ValueType> sum = Function.X.call( new Tsr[]{ this, ones } );
        return Function.DIV.call( new Tsr[]{ sum, new Tsr( this.size() ) } );
        //TODO :Function.DIV.call(new Tsr[]{sum, new Tsr(this.size())});
    }


    // ND-Iteration :
    //=========================

    @NotNull
    @Override
    public Iterator<ValueType> iterator()
    {
        NDIterator _ndi = NDIterator.of( this );
        return new Iterator<ValueType>() 
        {
            private int _count = 0;
            private final int _size = size();

            @Override
            public boolean hasNext() {
                return _count != _size;
            }

            @Override
            public ValueType next() {
                Object o = getValueAt( _ndi.i() );
                _ndi.increment();
                _count ++;
                return (ValueType) o;
            }
        };
    }


    //MODIFICATION :
    //=========================

    /**
     *
     * @param error A tensor which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<ValueType> backward( Tsr<ValueType> error ) {
        if ( !forComponent( GraphNode.class, node -> node.backward( error ) ) && this.rqsGradient() ) {
            addToGradient( error );
        }
        return this;
    }

    /**
     *  This method turns the given scalar value and
     *  turns it into a matching tensor (same shape)
     *  which will be backpropagated through the
     *  recorded computation graph.
     *
     * @param value A scalar which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<ValueType> backward( double value ) {
        backward( new Tsr( _conf.shape(), value ) );
        return this;
    }

    /**
     *  This method assumes that the user wants to backpropagate
     *  an error of "1" having the same shape as
     *  this tensor.
     *
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<ValueType> backward() {
        backward( 1 );
        return this;
    }

    public void applyGradient() {
        forComponent( JITProp.class, JITProp::execute );
        remove( JITProp.class );
        forComponent(
                Tsr.class,
                g -> {
                    forComponent( Optimizer.class, o -> o.optimize( this ) );
                    remove( Tsr.class );
                    boolean inlineSafety = Neureka.instance().settings().autograd().isPreventingInlineOperations();
                    if ( inlineSafety )
                        Neureka.instance().settings().autograd().setIsPreventingInlineOperations( false );
                    // INLINE OPERATION :
                    Function.Detached.PLUS_ASSIGN.call( new Tsr[]{ this, g } );
                    // INLINE END !
                    if ( inlineSafety )
                        Neureka.instance().settings().autograd().setIsPreventingInlineOperations( true );
                }
        );
    }

    public void detach() {
        this.remove( GraphNode.class );
    }

    // TENSOR OPERATION (OVERLOADABLE):
    //=================================

    public Tsr<ValueType> T() { // Transposed!
        StringBuilder operation = new StringBuilder();
        for ( int i = rank() - 1; i >= 0; i-- ) operation.append( i ).append( ( i == 0 ) ? "" : ", " );
        operation = new StringBuilder( "[" + operation + "]:(I[ 0 ])" );
        return new Tsr<>( this, operation.toString() );
    }

    public Tsr<ValueType> plus( Tsr<ValueType> other ) {
        return Function.PLUS.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> plusAssign( Tsr<ValueType> other ) {
        return Function.Detached.PLUS_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> plus( Double value ) {
        return plus( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> minus( Tsr<ValueType> other ) {
        return Function.MINUS.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> minusAssign( Tsr<ValueType> other ) {
        return Function.Detached.MINUS_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> negative(){
        return Function.NEG.call( new Tsr[]{ this } );
    }

    public Tsr<ValueType> multiply( Tsr<ValueType> other ) {
        return Function.MUL.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> timesAssign( Tsr<ValueType> other ) {
        return Function.Detached.MUL_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> multiply( Double value ) {
        return multiply( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> div( Tsr<ValueType> other ) {
        return Function.DIV.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> div( Double value ) {
        return div( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> divAssign( Tsr<ValueType> other ){
        return Function.Detached.DIV_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> mod( Tsr<ValueType> other ) {
        return Function.MOD.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> modAssign( Tsr<ValueType> other ) {
        return Function.Detached.MOD_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> power( Tsr<ValueType> other ) {
        return Function.POW.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> power( Double value ){
        return power( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> xor( Tsr<ValueType> other ) {
        return Function.POW.call( new Tsr[]{ this, other} );
    }

    public Tsr<ValueType> xor( Double value ) {
        return xor( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> dot( Tsr<ValueType> b ) {
        Tsr<ValueType> a = this;
        int[][] fitter = AbstractNDArray.Utility.Indexing.makeFit( a.getNDConf().shape(), b.getNDConf().shape() );
        boolean doReshape = false;
        for ( int i = 0; i < fitter[ 0 ].length && !doReshape; i++ ) if ( fitter[ 0 ][ i ] != i ) doReshape = true;
        for ( int i = 0; i < fitter[ 1 ].length && !doReshape; i++ ) if ( fitter[ 1 ][ i ] != i ) doReshape = true;
        if ( doReshape ) {
            a = Function.create( AbstractNDArray.Utility.Stringify.strConf( fitter[ 0 ] ) + ":(I[ 0 ])" ).call( a );
            b = Function.create( AbstractNDArray.Utility.Stringify.strConf( fitter[ 1 ] ) + ":(I[ 0 ])" ).call( b );
        }
        return Function.X.call( new Tsr[]{ a, b } ).dimtrim();
    }

    public Tsr<ValueType> dimtrim(){
        return Function.DIMTRIM.call( this );
    }

    public boolean isCase( Tsr<ValueType> t ) {
        boolean[] found = { false };
        this.forComponent( Relation.class, r -> r.foreachChild( c -> {
                if ( c.equals( t ) ) found[ 0 ] = true;
            }));
        return found[ 0 ];
    }

    public boolean contains( Tsr<ValueType> t ){
        return isCase( t );
    }

    public Tsr<ValueType> label( String[][] labels ) {
        IndexAlias indexAlias = find( IndexAlias.class );
        if ( indexAlias == null ) {
            indexAlias = new IndexAlias( this.rank() );
            add( indexAlias );
        }
        for( int i = 0; i < labels.length; i++ ) {
            if ( labels[ i ] != null ) {
                for ( int ii = 0; ii < labels[ i ].length; ii++ ) {
                    if ( labels[ i ][ ii ] != null ) indexAlias.set( i, labels[ i ][ ii ], ii );
                }
            }
        }
        return this;
    }

    public Tsr<ValueType> label( List<List<Object>> labels ) {
        IndexAlias indexAlias = find( IndexAlias.class );
        if ( indexAlias == null ) add( new IndexAlias( labels ) );
        return this;
    }

    public Tsr<ValueType> label( Map<Object, List<Object>> labels ) {
        this.add( new IndexAlias<>( labels, this ) );
        return this;
    }

    private void _putAtCheckFor( Tsr value ) {
        if ( value.isEmpty() ) {
            String message = "Provided tensor is empty! Empty tensors cannot be injected.";
            _LOGGER.error( message );
            throw new IllegalArgumentException( message );
        }
    }

    public Tsr<ValueType> putAt( List<?> key, Tsr<ValueType> value ) {
        _putAtCheckFor( value );
        Tsr<ValueType> slice = ( key == null ) ? this : (Tsr) getAt( key );
        return _putAt( slice, value );
    }

    public Tsr<ValueType> putAt( Map<?,?> key, Tsr<ValueType> value ) {
        _putAtCheckFor( value );
        Tsr<ValueType> slice = ( key == null ) ? this : (Tsr) getAt( key );
        return _putAt( slice, value );
    }

    private Tsr<ValueType> _putAt( Tsr<ValueType> slice, Tsr<ValueType> value )
    {
        boolean valueIsDeviceVisitor = false;
        if ( slice.isOutsourced() && !value.isOutsourced() ) {
            Device device = slice.find( Device.class );
            try {
                device.store( value );
            } catch ( Exception exce ) {
                _LOGGER.error( "Trying to migrate target slice tensor to device failed.", exce );
                throw exce;
            }
            valueIsDeviceVisitor = true;
        }
        if ( this.isEmpty() && slice.isEmpty() || slice.size() != value.size() ) _become( value ); // TODO: Rethink this a little
        else new Tsr( new Tsr[]{ slice, value }, "I[ 0 ] <- I[ 1 ]", false );
        try {
            if ( valueIsDeviceVisitor ) value.find( Device.class ).restore( value );
        } catch ( Exception exception ) {
            _LOGGER.error( "Trying to migrate source tensor back to original location failed.", exception );
            throw exception;
        }
        return this;
    }

    public double getAt( int[] idx ){
        return value64( i_of_idx( idx ) );
    }

    public Object getAt( Object i1, Object i2 ) {
        List<Object> args = Arrays.asList( i1, i2 );
        return getAt( args );
    }

    public Object getAt( int i ) {
        return getAt( Arrays.asList(_conf.idx_of_i( i )).toArray() );
    }

    public Object getAt( double i ) {
        return getAt( Arrays.asList( _conf.idx_of_i( (int) Math.floor( i ) ) ).toArray() );
    }

    public Object getAt( BigDecimal i ) {
        return getAt( Arrays.asList( _conf.idx_of_i(( i ).intValue()) ).toArray() );
    }

    public Object getAt( Map<?,?> rangToStrides )
    {
        if ( rangToStrides == null ) return this;
        int[] newOffset = new int[ this.rank() ]; // ...not a simple slice... Advanced:
        int[] newSpread = new int[ this.rank() ];
        int[] newShape = new int[ this.rank() ];
        Object[] ranges = rangToStrides.keySet().toArray();
        _configureSubsetFromRanges( ranges, newOffset, newSpread, newShape, 0 );
        Object[] steps = rangToStrides.values().toArray();
        for ( int i = 0; i < this.rank(); i++ ) {
            newSpread[ i ] = (Integer) steps[ i ];
            newShape[ i ] /= (Integer) steps[ i ];
        }
        return _sliceOf( newShape, newOffset, newSpread );
    }

    public Tsr shallowCopy()
    {
        if ( this.isEmpty() || this.isUndefined() ) return this;
        List<List<Integer>> ranges = new ArrayList<>();
        for ( int e : this.shape() ) {
            List<Integer> rangeAsList = new ArrayList<>();
            for ( int i = 0; i < e; i++ ) rangeAsList.add( i );
            ranges.add( rangeAsList);
        }
        return (Tsr) getAt( ranges.toArray() );
    }

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    public Object getAt( Object key ) {
        if ( key == null ) return this;
        if ( key instanceof Object[] && ((Object[]) key).length == 0 ) key = new ArrayList<>();
        if ( key instanceof List && ( (List<?>) key ).isEmpty() ) {
            /*
                An empty List implementation instance is being interpreted as
                the request to create an identical slice, meaning that the
                resulting tensor views the same data as its parent while not
                being the same instance. (In a sense, its a shallow copy!)
             */
            //if ( this.isEmpty() || this.isUndefined() ) return this;
            //for ( int e : this.shape() ) {
            //    List<Integer> rangeAsList = new ArrayList<>();
            //    for (int i = 0; i < e; i++) rangeAsList.add(i);
            //    ( (List<Object>) key ).add(rangeAsList);
            //}
            return shallowCopy();
        }

        int[] newOffset = new int[ this.rank() ];
        int[] newSpread = new int[ this.rank() ];
        int[] newShape = new int[ this.rank() ];
        key = ( key instanceof List ) ? ((List<?>) key).toArray() : key;

        if ( key instanceof Object[] ) {
            boolean allInt = true;
            for (Object o : (Object[]) key) allInt = allInt && o instanceof Integer;
            if ( allInt && ((Object[]) key).length == rank() ) {
                key = _intArray((Object[]) key);
                newOffset = (int[]) key;
                if ( key != null ) {
                    for ( int i = 0; i < this.rank(); i++ )
                        newOffset[i] = ( newOffset[i] < 0 ) ? _conf.shape( i ) + newOffset[ i ] : newOffset[ i ];
                    return IO.getFrom( this, newOffset );
                }
            } else {
                boolean hasScale = false;
                for ( Object o : (Object[]) key ) hasScale = hasScale || o instanceof Map;
                if ( allInt ) _configureSubsetFromRanges(
                        new Object[]{ _intArray( (Object[]) key ) },
                        newOffset, newSpread, //idxbase,
                        newShape,
                        0
                );
                else _configureSubsetFromRanges( (Object[]) key, newOffset, newSpread, newShape, 0 );
            }
        } else {
            String message = "Cannot create tensor slice from key of type '" + key.getClass().getName() + "'!";
            _LOGGER.error( message );
            throw new IllegalArgumentException( message );
        }
        return _sliceOf( newShape, newOffset, newSpread );
    }

    private Tsr _sliceOf( int[] newShape, int[] newOffset, int[] newSpread ) {
        this.setIsVirtual( false );
        Tsr<ValueType> subset = new Tsr<>();
        subset._value = this._value;
        subset._type = this._type;
        int[] newTranslation = this._conf.translation();
        int[] newIdxmap = NDConfiguration.Utility.newTlnOf( newShape );

        for ( int i = 0; i < this.rank(); i++ )
            newSpread[ i ] = ( newSpread[i] == 0 ) ? 1 : newSpread[ i ];

        for ( int i = 0; i < newOffset.length; i++ )
            newOffset[ i ] = newOffset[ i ] + getNDConf().offset( i ); // Offset is being inherited!

        Tsr<?> rootTensor = ( this.isSlice() ) ? find( Relation.class ).findRootTensor() : this;
        Tsr<?> parentTensor = ( this.isSlice() ) ? find( Relation.class ).getParent() : this;
        /*
            The following code check the validity of the slice shape ranges with
            respect to the 'parentTensor' of this new slice.
         */
        if ( parentTensor.rank() != newShape.length || rootTensor != parentTensor ) {
            // TODO! This requires some more thought about how to check this!
            // THIS CASE HAS NOT YET BEEN THOUGHT TROUGH!
        } else {
            /*
                1. We know that inside this else branch 'this' tensor is a first order slice!
                (So it is not a slice of a slice... reason : 'rootTensor == parentTensor' )

                2. There is however uncertainty about the 'true shape' of this parent tensor!
                Meaning : It might have been reshaped and could therefore be distorted with
                respect to the slice that is currently being prepared!
                -> This means we have to take this possible reshaping into account!
                Like so:

                The following uses an int array also called 'reshapeRelation'.
                This is simply the 'reshape array' which has been recorded inside the 'Relation' component
                by the 'Reshape' operation! ( Hopefully! :) )

                The following would occur when : "new Tsr(...).T().gatAt(...);"
                Transposing a tensor performs an inline reshaping of an identical
                slice of the original tensor! Then again slicing this tensor
                via the 'getAt(...)' method leads us to a situation where
                the following variable is NOT NULL! :
             */
            int[] reshaped = ( this.isSlice() ) ? parentTensor.find( Relation.class ).getReshapeRelationFor( this ) : null;
            reshaped = ( reshaped != null ) ? Reshape.invert( reshaped ) : null;
            for ( int i = 0; i < parentTensor.rank(); i++ ) {
                int ii = ( reshaped != null ) ? reshaped[ i ] : i;
                int top = newOffset[ i ] + newShape[ i ];
                if ( top > parentTensor.shape( ii ) ) {
                    String message =
                            "Cannot create slice because ranges are out of the bounds of the targeted tensor.\n" +
                                    "At index '" + i + "' : offset '" + newOffset[ i ] + "' + shape '" + newShape[ i ] + "' = '" + top + "',\n" +
                                    "which is larger than the target shape '" + parentTensor.shape( ii ) + "' at the same index!";
                    Exception exception = new IllegalArgumentException( message );
                    _LOGGER.error( message, exception );
                    throw new IllegalArgumentException( exception );
                }
            }
        }

        subset._conf = AbstractNDC.construct( newShape, newTranslation, newIdxmap, newSpread, newOffset );

        if ( this.isOutsourced() ) {
            Device device = this.find( Device.class );
            device.store( subset, this );
            subset.setIsOutsourced( true );
        }
        if ( this.isVirtual() ) subset.setIsVirtual( true );
        subset.add( new Relation().addParent( this ) );
        Relation parent = find( Relation.class );
        parent = ( parent != null ) ? parent : new Relation();
        parent.addChild( subset );
        this.add( parent );
        return subset;
    }

    /**
     *
     * @param ranges Elements of this array might be multiple things:
     *               - A map whose first entry represents a mapping between range and steps.
     *               - A list from which a first and last entry will be interpreted as range.
     *               - Any other object which might bew found in a 'IndexAlias' component.
     * @param offset Start index for every rank.
     * @param newShape New shape of the new sub-tensor.
     * @param iOffset Rank offset incremented according to recursive calls.
     * @return A new rank index.
     */
    private int _configureSubsetFromRanges(
            Object[] ranges,
            //int[] idxbase,
            int[] offset,  int[] spread,
            int[] newShape,
            int iOffset
    ) {
        for ( int i = 0; i < ranges.length; i++ ) {
            int first = 0;
            int last = 0;
            if( ranges[ i ] instanceof int[] ) {
                if ( ranges[ i ] instanceof int[] ) {
                    List<Integer> intList = new ArrayList<>( ( (int[]) ranges[ i ] ).length );
                    for ( int ii : (int[]) ranges[ i ] ) intList.add( ii );
                    ranges[ i ] = intList;
                }
            } else if ( ranges[ i ] instanceof String[] ) {
                if ( ranges[ i ] instanceof String[] ) {
                    List<String> strList = new ArrayList<>( ( (String[]) ranges[ i ] ).length);
                    for ( String ii : (String[]) ranges[ i ] ) strList.add( ii );
                    ranges[ i ] = strList;
                }
            }
            if ( !( ranges[ i ] instanceof  List ) ) {
                if ( ranges[ i ] instanceof Map ) {
                    Object[] ks = ( (Map<?,?>) ranges[ i ] ).keySet().toArray();
                    Object[] steps = ( (Map<?,?>) ranges[ i ]).values().toArray();
                    int newI = _configureSubsetFromRanges( ks, offset, spread, newShape, i + iOffset );
                    //for ( int ii = rank(); ii < ( rank() + steps.length ); ii++ ) {
                    //    idxbase[ ii + i + offset ] = (Integer) steps[ ii - rank() ];
                    //    newShape[ ii + i + offset - rank() ] /= idxbase[ ii + i + offset ];
                    //}
                    for ( int ii = 0; ii < steps.length; ii++ ) {
                        spread[ ii + i + iOffset ] = (Integer) steps[ ii ];
                        newShape[ ii + i + iOffset ] /= spread[ ii + i + iOffset ];
                    }
                    i = newI;
                    continue;
                } else if ( ranges[ i ] instanceof Integer ) {
                    first = (Integer) ranges[ i ];
                    last = (Integer) ranges[ i ];
                } else {
                    IndexAlias<?> indexAlias = find( IndexAlias.class );
                    if ( indexAlias != null ) {
                        int position = indexAlias.get( ranges[ i ], i + iOffset );
                        first = position;
                        last = position;
                    } else {
                        String message = "Given indexAlias key at axis " + ( i + iOffset ) + " not found!";
                        _LOGGER.error( message );
                        throw new IllegalStateException( message );
                    }
                }
            } else {
                ranges[ i ] = ( (List<?>) ranges[ i ] ).toArray();
                ranges[ i ] = ( ( (Object[]) ranges[ i ] )[ 0 ] instanceof List )
                        ? ( (List<?>) ( (Object[]) ranges[ i ] )[ 0 ] ).toArray()
                        : ( (Object[]) ranges[ i ] );
                if (
                        !( ( (Object[]) ( ranges[ i ] ) )[ 0 ] instanceof Integer )
                                || !( ( (Object[]) ranges[ i ] )[ ( (Object[]) ( ranges[ i ] ) ).length - 1 ] instanceof Integer )
                ) {
                    IndexAlias<?> indexAlias = find( IndexAlias.class );
                    if ( !( ( (Object[]) (ranges[ i ]) )[ 0 ] instanceof Integer ) ) {
                        if ( indexAlias != null ) {
                            first = indexAlias.get( ( (Object[]) ranges[ i ])[ 0 ], i + iOffset );
                        }
                    }
                    else first = (Integer) ( (Object[]) ranges[ i ] )[ 0 ];

                    if ( !( ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ] instanceof Integer )  ) {
                        if ( indexAlias != null ) {
                            last = indexAlias.get(
                                    ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ],
                                    i + iOffset
                            );
                        }
                    }
                    else last = (Integer) ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ];

                } else {
                    first = (Integer)( (Object[]) ranges[ i ] )[ 0 ];
                    last = (Integer) ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ];
                }
            }
            if ( first < 0 && last < 0 && first > last ) {
                int temp = first;
                first = last;
                last = temp;
            }
            first = ( first < 0 ) ? _conf.shape( i ) + first : first;
            last = ( last < 0 ) ? _conf.shape( i ) + last : last;
            newShape[ i + iOffset ] = ( last - first ) + 1;
            offset[ i + iOffset ] = first;
        }
        return ranges.length + iOffset - 1;
    }

    public static class IO
    {
        private IO(){}

        public static double getFrom( Tsr<?> t, int i ) {
            if ( t.isEmpty() || t.isUndefined() ) return 0;
            else if ( t.isVirtual() ) return t.value64()[ 0 ];
            return t.value64()[ t.i_of_i( i ) ];
        }

        public static double getFrom( Tsr<?> t, int[] idx ) {
            t.setIsVirtual( false );
            return t.value64()[ t.i_of_idx( idx ) ];
        }

        public static void setInto( Tsr<?> t, int i, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_i( i ) ] = value;
        }

        public static void setInto( Tsr<?> t, int[] idx, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_idx( idx ) ] = value;
        }

        public static void addInto( Tsr<?> t, int i, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_i( i ) ] += value;
        }

        public static void addInto( Tsr<?> t, int[] idx, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_idx( idx ) ] += value;
        }

        public static Tsr<?> addInto( Tsr<?> t, Tsr<?> source ) {
            if ( t.isVirtual() && source.isVirtual() ) t.value64()[ 0 ] += source.value64()[ 0 ];
            else FunctionBuilder.build( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])", false ).call( new Tsr[]{ t, source } );
            return source;
        }

        public static void subInto( Tsr<?> t, int i, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_i( i ) ] -= value;
        }

        public static void subInto( Tsr<?> t, int[] idx, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_idx( idx ) ] -= value;
        }

        public static void subInto( Tsr<?> t, Tsr<?> source ) {
            if ( t.isVirtual() && source.isVirtual() ) {
                t.value64()[ 0 ] -= source.value64()[ 0 ];
            } else {
                if ( t.isVirtual() ) t.setIsVirtual( false );
                int[] index = new int[ t.getNDConf().shape().length ];
                int size = t.size();
                for ( int i = 0; i < size; i++ ) {
                    IO.subInto( t, index, IO.getFrom( source, index ) );
                    NDConfiguration.Utility.increment( index, t.getNDConf().shape() );
                }
            }
        }

        public static void mulInto( Tsr<?> t, int i, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_i( i ) ] *= value;
        }

        public static void mulInto( Tsr<?> t, int[] idx, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_idx( idx ) ] *= value;
        }

    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    @Override
    public Object getValueAt( int i ) {
        if ( this.is32() ) return value32( i );
        else if ( this.is64() ) return value64( i );
        else if ( _value instanceof short[] ) return ( (short[]) _value )[ i ];
        else return ( (ValueType[]) _value )[ i ];
    }

    public Tsr<ValueType> setValue64( double[] value ) {
        if ( this.isOutsourced() ) this.find( Device.class ).overwrite64( this, value );
        else if ( _value == null ) {
            _value = value;
            _type = DataType.instance( F64.class );
        }
        else if ( _value instanceof float[] )
            for ( int i = 0; i < value.length; i++ ) ( (float[]) _value )[ i ] = (float) value[ i ];
        else if ( _value instanceof double[] )
            for ( int i = 0; i < value.length; i++ ) ( (double[]) _value )[ i ] = value[ i ];
        return this;
    }

    public Tsr<ValueType> setValue32( float[] value ) {
        if ( this.isOutsourced() ) this.find( Device.class ).overwrite32( this, value );
        else if ( _value == null ) {
            _value = value;
            _type = DataType.instance( F32.class );
        }
        else if ( _value instanceof float[] )
            for ( int i = 0; i < value.length; i++ ) ( (float[]) _value )[ i ] = value[ i ];
        else if ( _value instanceof double[] )
            for ( int i = 0; i < value.length; i++ ) ( (double[]) _value )[ i ] = value[ i ];
        return this;
    }

    public Tsr<ValueType> setValue( Object value ) {
        if ( value instanceof float[] ) this.setValue32( (float[]) value );
        else if ( value instanceof  double[] ) this.setValue64( (double[]) value );
        else if ( value instanceof Float ) {
            this.setIsVirtual( true );
            if ( this.is32() ) ( (float[]) _value )[ 0 ] = (Float) value;
            else ( (double[]) _value )[ 0 ] = ( (Float) value ).doubleValue();
        } else if ( value instanceof Double ) {
            this.setIsVirtual( true );
            if ( this.is64() ) ( (double[]) _value )[ 0 ] = (Double) value;
            else ( (float[]) _value )[ 0 ] = ( (Double) value ).floatValue();
        }
        return this;
    }

    public Object getValue() {
        if ( this.isOutsourced() ) {
            Device device = find( Device.class );
            return ( this.is32() )
                    ? device.value32f( this )
                    : device.value64f( this );
        }
        return _value;
    }

    public Object getData() {
        return _value;
    }

    public double[] gradient64() {
        Tsr<ValueType> gradient = this.find( Tsr.class );
        if ( gradient == null ) return new double[ 0 ];
        return ( this.is32() )
                ? DataConverter.Utility.floatToDouble( gradient.value32() )
                : gradient.value64();
    }

    public float[] gradient32() {
        Tsr<ValueType> gradient = this.find( Tsr.class );
        if ( gradient == null ) return new float[ 0 ];
        return ( this.is64() )
                ? DataConverter.Utility.doubleToFloat( gradient.value64() )
                : gradient.value32();
    }

    public Tsr<ValueType> addToGradient( Tsr<ValueType> error ) {
        if (
                !forComponent(
                    Tsr.class,
                        g ->
                        this.add(
                                Function.Detached.PLUS_ASSIGN.call( new Tsr[]{ g, error } )
                        )
                )
        ) add( error ).forComponent( Device.class, d -> {
            try {
                d.store( error ) ;
            } catch ( Exception exception ) {
                _LOGGER.error( "Failed trying to store a given error to a device for gradient accumulation.", exception );
                throw exception;
            }
        });
        return this;
    }

    public Tsr<ValueType> to32() {
        if ( this.is64() ) {
            Device device = this.find( Device.class );
            try {
                if ( device != null ) device.restore( this );
            } catch ( Exception exception ) {
                _LOGGER.error( "Failed to restore tensor from device for datatype migration.", exception );
                throw exception;
            }
            _value = DataConverter.Utility.doubleToFloat( (double[]) _value );
            forComponent( Tsr.class, Tsr::to32 );
            try {
                if ( device != null ) device.store( this );
            } catch ( Exception exception ) {
                _LOGGER.error( "Failed to store tensor back to original device after datatype migration.", exception );
                throw exception;
            }
        }
        return this;
    }

    public Tsr<ValueType> to64() {
        if ( this.is32() ) {
            Device device = this.find( Device.class );
            try {
                if ( device != null ) device.restore( this );
            } catch ( Exception exception ) {
                _LOGGER.error( "Failed to restore tensor from device for datatype migration.", exception );
                throw exception;
            }
            _value = DataConverter.Utility.floatToDouble( (float[]) _value );
            forComponent( Tsr.class, Tsr::to64 );
            try {
                if ( device != null ) device.store( this );
            } catch ( Exception exception ) {
                _LOGGER.error( "Failed to store tensor back to original device after datatype migration.", exception );
                throw exception;
            }
        }
        return this;
    }


    public <T> Tsr<T> asType( Class<T> typeClass )
    {
        Class<?> realTypeClass = typeClass;
        if ( typeClass == Double.class ) realTypeClass = F64.class;
        else if ( typeClass == Float.class ) realTypeClass = F32.class;
        else if ( typeClass == Integer.class ) realTypeClass = I32.class;
        else if ( typeClass == Short.class ) realTypeClass = I16.class;

        if ( this.isOutsourced() ) {
            _type = DataType.instance( realTypeClass );
            return (Tsr<T>) this;
        }
        // else conversion :
        DataType newDT = DataType.instance( realTypeClass );
        if (
                newDT.typeClassImplements( NumericType.class ) &&
                        _type.typeClassImplements( NumericType.class )
        ) {
            NumericType<?,Object> instance   = (NumericType<?, Object>) newDT.getTypeClassInstance();
            NumericType<?,Object> originType = (NumericType<?, Object>) _type.getTypeClassInstance();
            instance.convert( _value, originType.targetArrayType() );
        }
        return (Tsr<T>) this;
    }

    public double value64( int i ) {
        if ( this.isOutsourced() ) return find( Device.class ).value64f( this, i );
        if ( this.isVirtual() ) {
            if ( this.is64() ) return ( (double[]) _value )[ 0 ];
            else return ( (float[]) _value )[ 0 ];
        } else {
            if ( this.is64() ) return ( (double[]) _value )[ i ];
            else return ( (float[]) _value )[ i ];
        }
    }

    public double[] value64() {
        Device found = this.find( Device.class );
        if ( _value == null && this.isOutsourced() && found != null ) {
            return found.value64f( this );
        }
        double[] newValue = ( this.is64() )
                ? (double[]) _value
                : DataConverter.Utility.floatToDouble( (float[]) _value );
        if ( this.isVirtual() && newValue != null && this.size() > 1 ) {
            newValue = new double[ this.size() ];
            double[] value = ( this.is64() )
                    ? (double[]) _value
                    : DataConverter.Utility.floatToDouble( (float[]) _value );
            Arrays.fill( newValue, value[ 0 ] );
        }
        return newValue;
    }

    public float value32( int i ) {
        if ( this.isOutsourced() ) return find( Device.class ).value32f( this, i );
        if ( this.isVirtual() ) {
            if ( this.is64() ) return (float) ( (double[]) _value )[ 0 ];
            else return ( (float[]) _value )[ 0 ];
        } else {
            if ( this.is64() ) return (float) ( (double[]) _value )[ i ];
            else return ( (float[]) _value )[ i ];
        }
    }

    public float[] value32() {
        Device<ValueType> found = this.find( Device.class );
        if ( _value == null && this.isOutsourced() && found != null ) {
            return found.value32f( this );
        }
        float[] newValue = ( this.is64() )
                ? DataConverter.Utility.doubleToFloat( (double[]) _value )
                : (float[]) _value;
        if ( this.isVirtual() && newValue != null ) {
            newValue = new float[ this.size() ];
            Arrays.fill( newValue, newValue[ 0 ] );
        }
        return newValue;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //DISPLAY :
    //=========================
    public String toString( String mode ){
        return _toString( mode, ( mode.contains( "f" ) ) ? "    " : null );
    }

    protected String _toString( String mode, String deep ) {
        String base = ( deep == null ) ? "" : "\n" + deep;
        String delimiter = ( deep == null ) ? "" : "    ";
        String half = ( deep == null ) ? "" : "  ";
        String deeper = ( deep == null ) ? deep : deep + delimiter;
        int max = ( mode.contains( "s" ) ) ? 3 : 50;
        if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        StringBuilder strShape = new StringBuilder();
        int[] shape = _conf.shape();
        for ( int i = 0; i < shape.length; i++ ) {
            strShape.append( shape[ i ] );
            if ( i < shape.length - 1 ) strShape.append( "x" );
        }
        boolean compact = mode.contains( "c" );
        strShape = new StringBuilder(
                ( Neureka.instance().settings().view().isUsingLegacyView() )
                        ? "[" + strShape + "]"
                        : "(" + strShape + ")"
        );
        if ( mode.contains( "shape" ) || mode.contains( "shp" ) ) return strShape.toString();
        String asString = "";
        asString += _stringified( _value, compact, max );
        asString = strShape + (
                        ( Neureka.instance().settings().view().isUsingLegacyView() )
                                ? ":(" + asString + ")"
                                : ":[" + asString + "]"
                );
        if ( mode.contains("g") && ( this.rqsGradient() || this.hasGradient() ) ) {
            asString += ":g:";
            Tsr<ValueType> gradient = this.find( Tsr.class );
            if ( gradient != null )
                asString += gradient.toString( "c" ).replace( strShape + ":", "" );
            else
                asString += ( (Neureka.instance().settings().view().isUsingLegacyView() ) ? "(null)" : "[null]" );
        }
        if ( mode.contains( "r" ) && this.has( GraphNode.class ) && this.find( GraphNode.class ).size() > 0 ) {
            GraphNode<ValueType> node = this.find( GraphNode.class );
            AtomicReference<String> enclosed = new AtomicReference<>( "; " );
            node.forEachDerivative( ( t, agent ) -> {
                if ( agent.derivative() == null ) {
                    enclosed.set( enclosed.get() + "->d(null), " );
                } else {
                    enclosed.set(
                            enclosed.get() +
                            base + "=>d|[ " +
                            base + delimiter + agent.derivative()._toString( mode, deeper ) + " " +
                            base + half + "]|:t{ " +
                            base + delimiter + (
                                    ( t.getPayload() != null ) ? t.getPayload()._toString( mode, deeper ) : t.toString("")
                            ) + " " +
                            base + half + "}, "
                    );
                }
            });
            asString += enclosed.get();
        }
        if ( mode.contains( "d" ) && this.has( GraphNode.class ) && this.find( GraphNode.class ).size() > 0 ) {
            GraphNode<ValueType> node = this.find( GraphNode.class );
            if ( node.mode() != 0 ) {
                AtomicReference<String> asAR = new AtomicReference<>( "; " );
                node.forEachDerivative( ( t, agent ) -> {
                    if ( agent.derivative() == null ) asAR.set( asAR.get() + "->d(" + agent.toString() + "), " );
                    else asAR.set( asAR.get() + "->d" + agent.derivative()._toString( mode, deeper ) + ", " );
                });
                asString += asAR.get();
            }
        }
        return asString;
    }

    private String _stringified(
            Object v,
            boolean format,
            int max
    ) {
        if ( v instanceof double[] ) return _stringified(
                i -> ( format )
                        ? Utility.Stringify.formatFP( ( (double[]) v )[ i ])
                        : String.valueOf( ( (double[] ) v )[ i ] ),
                max
        );
        else if ( v instanceof float[] ) return _stringified(
                i -> ( format )
                        ? Utility.Stringify.formatFP( ( (float[]) v )[ i ] )
                        : String.valueOf( ( (float[]) v )[ i ] ),
                max
        );
        else if ( v instanceof short[] )  return _stringified(
                i -> ( format )
                        ? Utility.Stringify.formatFP( ( (short[]) v )[ i ] )
                        : String.valueOf( ( (short[]) v )[ i ] ),
                max
        );
        else if ( v == null ) return _stringified(
                    i -> ( format )
                            ? Utility.Stringify.formatFP( value64( i ) )
                            : String.valueOf( value64( i ) ),
                    max
            );
        else return _stringified(
                    i -> String.valueOf( ( (Object[]) v )[ i ] ),
                    max
            );
    }

    private String _stringified(
            IntFunction<String> getter,
            int max
    ){
        StringBuilder asString = new StringBuilder();
        int size = this.size();
        int trim = ( size - max );
        size = ( trim > 0 ) ? max : size;
        for ( int i = 0; i < size; i++ ) {
            String vStr = getter.apply( ( this.isVirtual() ) ? 0 : i_of_i( i ) );
            asString.append( vStr );
            if ( i < size - 1 ) asString.append( ", " );
            else if ( trim > 0 ) asString.append( ", ... + " ).append( trim ).append( " more" );
        }
        return asString.toString();
    }

    public String toString() {
        return toString( "dgc" );
    }


    public static void makeFit( Tsr[] tsrs, boolean doesAD ) {
        int largest = -1;
        int[] shape = null;
        for ( Tsr t : tsrs ) if ( t.rank() > largest ) {
            largest = t.rank();
            shape = t.getNDConf().shape();
        }
        int prefix = 0;
        for ( int s : shape ) if ( s == 1 ) prefix++; else break;
        int postfix = 0;
        for ( int i = shape.length-1; i>=0; i-- ) if ( shape[ i ] == 1 ) postfix++; else break;
        for ( int i = 0; i < tsrs.length; i++ ) {
            if ( tsrs[ i ].rank() != largest ) {
                int[] oldShape = tsrs[ i ].getNDConf().shape();
                int[] newReshape = new int[ largest ];
                int padding = largest - oldShape.length;

                int handle = ( postfix <= prefix ) ? padding : largest - padding;
                for ( int ii = 0; ii < handle; ii++ ) newReshape[ ii ] = ( postfix <= prefix ) ? -1 : ii;
                for ( int ii = handle; ii < largest; ii++ ) newReshape[ ii ] = ( postfix <= prefix ) ? ii - padding : -1;

                Function f = Function.create(
                    AbstractNDArray.Utility.Stringify.strConf( newReshape ) + ":(I[ 0 ])",
                        doesAD
                );
                tsrs[ i ] = f.call( tsrs[ i ] );
            }
        }

    }

    public static class Create
    {
        private Create(){}

        public  static Tsr E( int[] shape ){
            return new Tsr( shape, 2.7182818284590452353602874713527 );
        }

        public static Tsr newRandom( int[] shape ){
            return newRandom( shape, 8701252152903546L );
        }

        public static Tsr newRandom( int[] shape, long seed ) {
            int size = NDConfiguration.Utility.szeOfShp( shape );
            return new Tsr( shape, DataConverter.Utility.newSeededDoubleArray( seed, size ) );
        }

        public static Tsr newTsrLike( Tsr template, double value ) {
            Tsr t = _newEmptyLike( template );
            if ( template.is32() ) t.setValue( (float) value );
            else t.setValue( value );
            try {
                if ( template.isOutsourced() ) ( (Device<Object>) template.find( Device.class ) ).store( t );
            } catch ( Exception exception ) {
                _LOGGER.error( "Failed storing a newly created tensor from a template tensor to its host device.", exception );
                throw exception;
            }
            return t;
        }

        public static Tsr newTsrLike( Tsr template ) { // The output tensor will not have gradients!
            Tsr t = _newEmptyLike( template );
            if ( template.is32() ) t.setValue32( new float[ template.size() ] );
            else t.setValue64( new double[ template.size() ] );
            try {
                if ( template.isOutsourced() ) ( (Device<Object>) template.find( Device.class ) ).store( t );
            } catch ( Exception exception ) {
                _LOGGER.error( "Failed storing a newly created tensor from a template tensor to its host device.", exception );
                throw exception;
            }
            return t;
        }

        private static Tsr _newEmptyLike( Tsr template ) {
            Tsr t = new Tsr();
            t._configureFromNewShape( template.getNDConf().shape(), false );
            return t;
        }

    }



    @Override
    public void update( Tsr<ValueType> oldOwner, Tsr<ValueType> newOwner ) {
        // This is means that this tensor is a gradient that is being
        // transferred to another tensor to serve as gradient...
        // No update task needs to occur. (This might change in the future...)
    }


}
