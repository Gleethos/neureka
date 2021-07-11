package neureka.utility;

import neureka.dtype.DataType;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ArrayUtils {

    private static Logger _LOG = LoggerFactory.getLogger(ArrayUtils.class);

    public interface Construction {
        void setType( DataType<?> type );
        void setConf( NDConfiguration conf );
        void setData( Object o );
        void allocate( int size );
        Object getData();
    }

    private final Construction _construction;

    public ArrayUtils(Construction construction) { this._construction = construction; }

    /**
     *  This method is responsible for instantiating and setting the _conf variable.
     *  The core requirement for instantiating {@link NDConfiguration} interface implementation s
     *  is a shape array of integers which is being passed to the method... <br>
     *  <br>
     *
     * @param newShape An array if integers which are all greater 0 and represent the tensor dimensions.
     */
    public void configureFromNewShape( int[] newShape, boolean makeVirtual, boolean autoAllocate )
    {
        int size = NDConfiguration.Utility.szeOfShp( newShape );
        if ( size == 0 ) {
            String shape = Arrays.stream( newShape ).mapToObj( String::valueOf ).collect( Collectors.joining( "x" ) );
            String message = "The provided shape '"+shape+"' must not contain zeros. Dimensions lower than 1 are not possible.";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        if ( _construction.getData() == null && autoAllocate ) _construction.allocate( size );
        //int length = _dataLength();
        //if ( length >= 0 && size != length && ( !this.isVirtual() || !makeVirtual) ) {
        //    String message = "Size of shape does not match stored data array size!";
        //    _LOG.error( message );
        //    throw new IllegalArgumentException( message );
        //}
        if ( makeVirtual ) _construction.setConf( VirtualNDConfiguration.construct( newShape ) );
        else {
            int[] newTranslation = NDConfiguration.Utility.newTlnOf( newShape );
            int[] newSpread = new int[ newShape.length ];
            Arrays.fill( newSpread, 1 );
            int[] newOffset = new int[ newShape.length ];
            _construction.setConf(
                    AbstractNDC.construct(
                            newShape,
                            newTranslation,
                            newTranslation, // indicesMap
                            newSpread,
                            newOffset
                    )
            );
        }
    }

    public  <V> void construct( int[] shape, V[] value ) {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        if ( size != value.length ) {
            _fromRange( shape, value );
        }
        else _construction.setData( value );
        configureFromNewShape( shape, false, true );
    }

    private <V> void _fromRange(int[] shape, V[] value ) {
        Class<?> givenClass = value[ 0 ].getClass();
        @SuppressWarnings("unchecked")
        final V[] newValue = (V[]) Array.newInstance(
                givenClass,
                NDConfiguration.Utility.szeOfShp( shape )
        );
        for ( int i = 0; i < newValue.length; i++ ) newValue[ i ] = value[ i % value.length ];
        _construction.setType( DataType.of( givenClass ) );
        _construction.setData( newValue );
    }

    public static Object optimizeArray( DataType<?> dataType, Object data, int size ) {
        if      ( data instanceof Integer[] ) return DataConverter.instance().convert( (Integer[]) data, int[].class,    size );
        else if ( data instanceof Double[]  ) return DataConverter.instance().convert( (Double[])  data, double[].class, size );
        else if ( data instanceof Float[]   ) return DataConverter.instance().convert( (Float[])   data, float[].class,  size );
        else if ( data instanceof Long[]    ) return DataConverter.instance().convert( (Long[])    data, long[].class,   size );
        else if ( data instanceof Short[]   ) return DataConverter.instance().convert( (Short[])   data, short[].class,  size );
        else if ( data instanceof Byte[]    ) return DataConverter.instance().convert( (Byte[])    data, byte[].class,   size );
        else if ( data instanceof Object[] )
            return ArrayUtils.optimizeObjectArray(dataType, (Object[]) data, size);
        else
            return data;
    }

    public static Object optimizeObjectArray( DataType<?> dataType, Object[] values, int size ) {
        Object data = values;
        IntStream indices = IntStream.iterate( 0, i -> i + 1 ).limit(size).map(i -> i % values.length );
        if      ( dataType == DataType.of(Double.class)  ) data = indices.mapToDouble( i -> (Double) values[i] ).toArray();
        else if ( dataType == DataType.of(Integer.class) ) data = indices.map( i -> (Integer) values[i] ).toArray();
        else if ( dataType == DataType.of(Long.class)    ) data = indices.mapToLong( i -> (Long) values[i] ).toArray();
        else if ( dataType == DataType.of(Float.class)   ) {
            float[] floats = new float[size];
            for( int i = 0; i < size; i++ ) floats[ i ] = (Float) values[ i % values.length ];
            data = floats;
        }
        else if ( dataType == DataType.of(Byte.class) ) {
            byte[] bytes = new byte[size];
            for( int i = 0; i < size; i++ ) bytes[ i ] = (Byte) values[ i % values.length ];
            data = bytes;
        }
        else if ( dataType == DataType.of(Short.class) ) {
            short[] shorts = new short[size];
            for( int i = 0; i < size; i++ ) shorts[ i ] = (Short) values[ i % values.length ];
            data = shorts;
        } else if ( values.length != size ) {
            Object[] objects = new Object[size];
            for( int i = 0; i < size; i++ ) objects[ i ] = values[ i % values.length ];
            data = objects;
        }
        return data;
    }


}
