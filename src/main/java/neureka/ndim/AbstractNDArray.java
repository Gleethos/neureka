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

            _         _                  _   _   _ _____
      /\   | |       | |                | | | \ | |  __ \   /\
     /  \  | |__  ___| |_ _ __ __ _  ___| |_|  \| | |  | | /  \   _ __ _ __ __ _ _   _
    / /\ \ | '_ \/ __| __| '__/ _` |/ __| __| . ` | |  | |/ /\ \ | '__| '__/ _` | | | |
   / ____ \| |_) \__ \ |_| | | (_| | (__| |_| |\  | |__| / ____ \| |  | | | (_| | |_| |
  /_/    \_\_.__/|___/\__|_|  \__,_|\___|\__|_| \_|_____/_/    \_\_|  |_|  \__,_|\__, |
                                                                                  __/ |
                                                                                |___/


*/

package neureka.ndim;

import neureka.Neureka;
import neureka.Tsr;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.*;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;
import neureka.utility.ArrayUtils;
import neureka.utility.DataConverter;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.stream.Collectors;


/**
 *  This is the precursor class to the final Tsr class from which
 *  tensor instances can be created. <br>
 *  The inheritance model of a tensor is structured as follows: <br>
 *  Tsr inherits from AbstractNDArray which inherits from AbstractComponentOwner
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 */
public abstract class AbstractNDArray<InstanceType, ValType> extends AbstractComponentOwner<InstanceType> implements Iterable<ValType>
{
    public interface Initializer<T> { T init(int i, int[] index );  }

    /**
     *  An interface provided by sl4j which enables a modular logging backend!
     */
    public static Logger _LOG; // Why is this not final ? -> For unit testing!

    private NDConfiguration _NDConf;

    private DataType<?> _dataType = DataType.of( Neureka.get().settings().dtype().getDefaultDataTypeClass() );

    private Object _data;

    public NDConfiguration getNDConf() { return this._NDConf; }

    public DataType<?> getDataType() { return this._dataType; }

    public Object getData() { return this._data; }

    public Class<?> getValueClass() { return ( _dataType != null ? _dataType.getTypeClass() : null ); }

    /**
     *  This method enables modifying the data-type configuration of this NDArray.
     *  Warning! The method should not be used unless absolutely necessary.
     *  This is because it can cause unpredictable inconsistencies between the
     *  underlying DataType instance of this NDArray and the actual type of the actual
     *  data it is wrapping (or it is referencing on a Device).<br>
     *  <br>
     * @param dataType The new dataType which ought to be set.
     * @return The final instance type of this class which enables method chaining.
     */
    public InstanceType setDataType( DataType<?> dataType )
    {
        if ( _data != null ) {
            String message = "Data type of tensor can only be set when data attribute is null!\n" +
                    "This is due to construction-consistency reasons.\n";
            throw new IllegalStateException( message );
        }
        _dataType = dataType;
        return (InstanceType) this;
    }

    protected void _setData( Object data )
    {
        if ( _dataType == null ) {
            String message = "Trying to set data in a tensor which does not have a DataTyp instance.";
            _LOG.error( message );
            throw new IllegalStateException( message );
        }
        if ( data != null && _dataType.typeClassImplements( NumericType.class ) ) {
            NumericType<?,?,?,?> numericType = (NumericType<?,?,?,?>) _dataType.getTypeClassInstance();
            if ( numericType.targetArrayType() != data.getClass() ) {
                String message = "Cannot set data whose type does not match what is defined by the DataType instance.\n" +
                        "Current type '"+numericType.targetArrayType().getSimpleName()+"' does not match '"+ data.getClass().getSimpleName()+"'.\n";
                _LOG.error( message );
                throw new IllegalStateException( message );
            }
        }
        _data = data;
    }

    protected <T> void _initData( Tsr.Initializer<T> initializer )
    {
        Object data = getData();
        if ( data instanceof double[] )
            for ( int i = 0; i < ( (double[]) data ).length; i++ )
                ( (double[]) data )[ i ] = (double) initializer.init( i, _NDConf.indicesOfIndex( i )  );
        else if ( data instanceof float[] )
            for ( int i = 0; i < ( (float[]) data ).length; i++ )
                ( (float[]) data )[ i ] = (float) initializer.init( i, _NDConf.indicesOfIndex( i )  );
        else if ( data instanceof int[] )
            for ( int i = 0; i < ( (int[]) data ).length; i++ )
                ( (int[]) data )[ i ] = (int) initializer.init( i, _NDConf.indicesOfIndex( i )  );
        else if ( data instanceof short[] )
            for ( int i = 0; i < ( (short[]) data ).length; i++ )
                ( (short[]) data )[ i ] = (short) initializer.init( i, _NDConf.indicesOfIndex( i )  );
        else if ( data instanceof byte[] )
            for ( int i = 0; i < ( (byte[]) data ).length; i++ )
                ( (byte[]) data )[ i ] = (byte) initializer.init( i, _NDConf.indicesOfIndex( i )  );
        else
            for ( int i = 0; i < ( (Object[]) data ).length; i++ )
                ( (Object[]) data )[ i ] = initializer.init( i, _NDConf.indicesOfIndex( i )  );
    }

    /**
     *  This method is responsible for allocating the data of this nd-array.
     *  It is protected and located in this abstract class so that a high degree of encapsulation
     *  is ensured for such crucial procedures like the allocation of the right data. <br>
     *  The actual allocation takes place inside an instance of the DataType class.
     *  This is because the data type has to be known in order to correctly perform an allocation.<br>
     *  <br>
     *
     * @param size The size of the data array which ought to be allocated.
     */
    protected void _allocate( int size ) { _data = _dataType.allocate( size ); }

    public abstract InstanceType setIsVirtual( boolean isVirtual );

    public abstract boolean isVirtual();

    protected boolean _constructAllFromOne( int[] shape, Object data ) {
        if ( data instanceof Double  ) { _constructAllF64( shape, (Double)  data ); return true; }
        if ( data instanceof Float   ) { _constructAllF32( shape, (Float)   data ); return true; }
        if ( data instanceof Integer ) { _constructAllI32( shape, (Integer) data ); return true; }
        if ( data instanceof Short   ) { _constructAllI16( shape, (Short)   data ); return true; }
        if ( data instanceof Byte    ) { _constructAllI8(  shape, (Byte)    data ); return true; }
        return false;
    }

    protected void _constructAllF64( int[] shape, double value ) {
        _constructAll( shape, F64.class );
        ( (double[]) getData())[ 0 ] = value;
    }

    private void _constructAllF32( int[] shape, float value ) {
        _constructAll( shape, F32.class );
        ( (float[]) getData())[ 0 ] = value;
    }

    private void _constructAllI32( int[] shape, int value ) {
        _constructAll( shape, I32.class );
        ( (int[]) getData())[ 0 ] = value;
    }

    private void _constructAllI16( int[] shape, short value ) {
        _constructAll( shape, I16.class );
        ( (short[]) getData())[ 0 ] = value;
    }

    private void _constructAllI8( int[] shape, byte value ) {
        _constructAll( shape, I8.class );
        ( (byte[]) getData())[ 0 ] = value;
    }

    private void _constructAll( int[] shape, Class<?> typeClass ) {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( typeClass ) );
        _allocate( 1 );
        setIsVirtual( size > 1 );
        _configureFromNewShape( shape, size > 1, true );
    }

    /**
     *  This method is responsible for instantiating and setting the _conf variable.
     *  The core requirement for instantiating {@link NDConfiguration} interface implementation s
     *  is a shape array of integers which is being passed to the method... <br>
     *  <br>
     *
     * @param newShape An array if integers which are all greater 0 and represent the tensor dimensions.
     */
    protected void _configureFromNewShape( int[] newShape, boolean makeVirtual, boolean autoAllocate )
    {
        int size = NDConfiguration.Utility.szeOfShp( newShape );
        if ( size == 0 ) {
            String shape = Arrays.stream( newShape ).mapToObj( String::valueOf ).collect( Collectors.joining( "x" ) );
            String message = "The provided shape '"+shape+"' must not contain zeros. Dimensions lower than 1 are not possible.";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        if ( getData() == null && autoAllocate ) _allocate( size );
        int length = _dataLength();
        if ( length >= 0 && size != length && ( !this.isVirtual() || !makeVirtual) ) {
            String message = "Size of shape does not match stored value64!";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        if ( makeVirtual ) setNDConf( VirtualNDConfiguration.construct( newShape ) );
        else {
            int[] newTranslation = NDConfiguration.Utility.newTlnOf( newShape );
            int[] newSpread = new int[ newShape.length ];
            Arrays.fill( newSpread, 1 );
            int[] newOffset = new int[ newShape.length ];
            setNDConf(
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

    private int _dataLength()
    {
        if ( !(getData() instanceof float[]) && !(getData() instanceof double[]) ) {
            if ( getData() instanceof Object[] ) return ((Object[]) getData()).length;
            else return -1;
        }
        else if ( getData() instanceof double[] ) return ( (double[]) getData()).length;
        else return ( (float[]) getData()).length;
    }


    protected void _constructForDoubles( int[] shape, double[] value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( F64.class ) );
        if ( size != value.length ) {
            _allocate( size );
            for ( int i = 0; i < size; i++ ) ( (double[]) getData())[ i ]  = value[ i % value.length ];
        }
        else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    protected void _constructForFloats( int[] shape, float[] value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( F32.class ) );
        if ( size != value.length ) {
            _allocate( size );
            for ( int i = 0; i < size; i++ ) ( (float[]) getData())[ i ]  = value[ i % value.length ];
        } else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    private void _constructForInts( int[] shape, int[] value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( I32.class ) );
        if ( size != value.length ) {
            _allocate( size );
            for ( int i = 0; i < size; i++ ) ( (int[]) getData())[ i ]  = value[ i % value.length ];
        } else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    private void _constructForShorts( int[] shape, short[] value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( I16.class ) );
        if ( size != value.length ) {
            _allocate( size );
            for ( int i = 0; i < size; i++ ) ( (short[]) getData())[ i ]  = value[ i % value.length ];
        } else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    private <V> void _constructForRange( int[] shape, DataType<?> dataType, V[] range )
    {
        if ( range.length != 0 && !( range[ 0 ] instanceof Number ) ) {
            Class<?> givenClass = range[ 0 ].getClass();
            @SuppressWarnings("unchecked")
            final V[] value = (V[]) Array.newInstance(
                    givenClass,
                    NDConfiguration.Utility.szeOfShp( shape )
            );
            for ( int i = 0; i < value.length; i++ ) value[ i ] = range[ i % range.length ];
            setDataType( DataType.of( givenClass ) );
            _setData( value );
            _construct( shape, value );
        } else {
            setDataType( dataType );
            if ( dataType.getTypeClass() == F64.class )
                _constructForDoubles(
                        shape,
                        DataConverter.Utility.objectsToDoubles( range, NDConfiguration.Utility.szeOfShp( shape ) )
                );
            else if ( dataType.getTypeClass() == F32.class  )
                _constructForFloats(
                        shape,
                        DataConverter.Utility.objectsToFloats( range, NDConfiguration.Utility.szeOfShp( shape ) )
                );
            else if ( dataType.getTypeClass() == I32.class )
                _constructForInts(
                        shape,
                        DataConverter.Utility.objectsToInts( range, NDConfiguration.Utility.szeOfShp( shape ) )
                );
            else if ( dataType.getTypeClass() == I16.class )
                _constructForShorts(
                        shape,
                        DataConverter.Utility.objectsToShorts( range, NDConfiguration.Utility.szeOfShp( shape ) )
                );
        }
    }

    private <V> void _construct( int[] shape, V[] value ) {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        if ( size != value.length ) {
            Class<?> givenClass = value[ 0 ].getClass();
            @SuppressWarnings("unchecked")
            final V[] newValue = (V[]) Array.newInstance(
                    givenClass,
                    NDConfiguration.Utility.szeOfShp( shape )
            );
            for ( int i = 0; i < newValue.length; i++ ) newValue[ i ] = value[ i % value.length ];
            setDataType( DataType.of( givenClass ) );
            _setData( newValue );
        }
        else _setData( value );
        _configureFromNewShape( shape, false, true );
    }


    protected void _tryConstructing( int[] shape, DataType<?> dataType, Object data ) {
        int size = NDConfiguration.Utility.szeOfShp(shape);
        if ( data instanceof List<?> ) {
            List<?> range = (List<?>) data;
            if ( dataType == DataType.of(Object.class) ) {
                // Nested Groovy list should be unpacked:
                if ( range.size() == 1 && range.get( 0 ).getClass().getSimpleName().equals("IntRange") )
                    range = (List<?>) range.get( 0 );
                _constructForRange(
                        shape,
                        DataType.of( F64.class ),
                        range.toArray()
                );
                return;
            }
            else
                data = range.toArray();
        }
        if ( data instanceof Object[] && DataType.of(((Object[])data)[0].getClass()) != dataType ) {
            for ( int i = 0; i < ( (Object[]) data ).length; i++ ) {
                ( (Object[]) data )[i] = DataConverter.instance().convert( ( (Object[]) data )[i], dataType.getJVMTypeClass() );
            }
            data = ArrayUtils.optimizeObjectArray(dataType, (Object[]) data, size);
        }
        if ( dataType == DataType.of( data.getClass() ) ) { // This means that "data" is a single value!
            if ( _constructAllFromOne( shape, data ) ) return;
        }
        else
            data = ArrayUtils.optimizeArray( dataType, data, size );

        setDataType( dataType );
        _configureFromNewShape( shape, false, false );
        _setData( data );
    }

    /**
     *  A virtual NDArray (tensor) is the opposite to an actual one. <br>
     *  Virtual means that the size of the underlying data does not match the real size of the NDArray.
     *  This is the case when the NDArray is filled with one element homogeneously.
     *  An example would be an all zeros array.<br>
     *  The reasoning behind this feature is memory efficiency.
     *  It would be unreasonable to allocate an arrays filled entirely with one and the same value item!
     *  <br>
     */
    protected void _virtualize()
    {
        _data = _dataType.virtualize(_data);
    }

    /**
     *  An actual NDArray (tensor) is the opposite to a virtual one. <br>
     *  Virtual means that the size of the underlying data does not match the real size of the NDArray.
     *  This is the case when the NDArray is filled with one element homogeneously.
     *  An example would be an all zeros array. The reasoning behind this feature is memory efficiency.
     *  It would be unreasonable to allocate an arrays filled entirely with one and the same value item!<br>
     *  <br>
     *  This method turns the data of a virtual NDArray into a newly allocated data array matching the
     *  size of the nd-array type... <br>
     */
    protected void _actualize()
    {
        _data = _dataType.actualize(_data, this.size() );
    }

    protected Object _convertedDataOfType( Class<?> typeClass )
    {
        DataType<?> newDT = DataType.of( typeClass );
        if (
                newDT.typeClassImplements( NumericType.class ) &&
                        getDataType().typeClassImplements( NumericType.class )
        ) {
            NumericType<?,Object, ?, Object> targetType  = (NumericType<?, Object,?, Object>) newDT.getTypeClassInstance();
            return targetType.readForeignDataFrom( iterator(), this.size() );
        }
        else
            return DataConverter.instance().convert( getData(), newDT.getTypeClass() );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public void forEach( Consumer<? super ValType> action ) {
        for ( ValType v : this ) action.accept( v );
    }

    @Override
    public Spliterator<ValType> spliterator()
    {
        return new Spliterator<ValType>()
        {
            @Override
            public boolean tryAdvance( Consumer<? super ValType> action ) {
                return false;
            }

            @Override
            public Spliterator<ValType> trySplit() {
                return null;
            }

            @Override
            public long estimateSize() {
                return 0;
            }

            @Override
            public int characteristics() {
                return 0;
            }
        };
    }

    /**
     *  An NDArray implementation ought to have some way to access its underlying data array.
     *  This method simple returns an element within this data array sitting at position "i".
     * @param i The position of the targeted item within the raw data array of an NDArray implementation.
     * @return The found object sitting at the specified index position.
     */
    public abstract Object getDataAt( int i );

    /**
     *  An NDArray implementation ought to have some way to selectively modify its underlying data array.
     *  This method simply returns an element within this data array sitting at position "i".
     * @param i The index of the data array entry which ought to be addressed.
     * @param o The object which ought to be placed at the requested position.
     * @return This very tensor in order to enable method chaining.
     */
    public abstract InstanceType setDataAt( int i, ValType o );

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  This method compares the passed class with the underlying data-type of this NDArray.
     *  If the data-type of this NDArray is equivalent to the passed class then the returned
     *  boolean will be true, otherwise the method returns false.
     *
     * @param typeClass The class which ought to be compared to the underlying data-type of this NDArray.
     * @return The truth value of the question: Does this NDArray implementation hold the data of the passed type?
     */
    public boolean is( Class<?> typeClass ) {
        DataType<?> type = DataType.of( typeClass );
        return type == _dataType;
    }

    public boolean is64() {
        return _data instanceof double[];
    }

    public boolean is32() {
        return _data instanceof float[];
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public int indexOfIndex( int i ) {
        return _NDConf.indexOfIndex( i );
    }

    public int[] IndicesOfIndex( int index ) {
        return _NDConf.indicesOfIndex( index );
    }

    public int indexOfIndices( int[] indices ) {
        return _NDConf.indexOfIndices(indices);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  This method sets the NDConfiguration of this NDArray.
     *  Therefore, it should not be used lightly as it can cause major internal inconsistencies.
     *
     * @param ndConfiguration The new NDConfiguration instance which ought to be set.
     * @return The final instance type of this class which enables method chaining.
     */
    public InstanceType setNDConf( NDConfiguration ndConfiguration )
    {
        if ( _NDConf != null && ndConfiguration != null ) {
            int s1 = Arrays.stream( _NDConf.shape() ).map( Math::abs ).reduce( 1, ( a, b ) -> a*b );
            int s2 = Arrays.stream( ndConfiguration.shape() ).map( Math::abs ).reduce( 1, ( a, b ) -> a*b );
            assert s1 == s2;
        }
        _NDConf = ndConfiguration;
        return (InstanceType) this;
    }

    //---

    public int rank() {
        return _NDConf.shape().length;
    }

    public List<Integer> shape() {
        return _asList(_NDConf.shape());
    }

    public int shape( int i ) {
        return _NDConf.shape()[ i ];
    }

    public List<Integer> indicesMap() {
        return _asList(_NDConf.indicesMap());
    }

    public List<Integer> translation() {
        return _asList(_NDConf.translation());
    }

    public List<Integer> spread() {
        return _asList(_NDConf.spread());
    }

    public List<Integer> offset() {
        return _asList(_NDConf.offset());
    }

    public int size() {
        return NDConfiguration.Utility.szeOfShp(_NDConf.shape());
    }

    protected static List<Integer> _asList( int[] array ) {
        List<Integer> intList = new ArrayList<>( array.length );
        for ( int i : array ) intList.add( i );
        return intList;
    }



    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    /**
     *  Static utility methods for the NDArray.
     */
    public static class Utility
    {
        public static class Stringify
        {
            @Contract( pure = true )
            public static String strConf( int[] conf ) {
                StringBuilder str = new StringBuilder();
                for ( int i = 0; i < conf.length; i++ )
                    str.append(conf[ i ]).append((i != conf.length - 1) ? ", " : "");
                return "[" + str + "]";
            }
        }

        /**
         * Indexing methods.
         */
        public static class Indexing
        {
            @Contract(pure = true)
            public static void shpCheck( int[] newShp, Tsr t ) {
                if ( NDConfiguration.Utility.szeOfShp(newShp) != t.size() ) {
                    throw new IllegalArgumentException(
                            "New shape does not match tensor size!" +
                                    " (" +
                                        Utility.Stringify.strConf(newShp) +
                                        ((NDConfiguration.Utility.szeOfShp(newShp) < t.size()) ? "<" : ">") +
                                        Utility.Stringify.strConf(t.getNDConf().shape()) + "" +
                                    ")"
                    );
                }
            }

            @Contract(pure = true)
            public static int[][] makeFit( int[] sA, int[] sB ) {
                int lastIndexOfA = 0;
                for ( int i = sA.length-1; i >= 0; i-- ) {
                    if (sA[ i ]!=1) {
                        lastIndexOfA = i;
                        break;
                    }
                }
                int firstIndexOfB = 0;
                for ( int i=0; i<sB.length; i++ ) {
                    if (sB[ i ]!=1) {
                        firstIndexOfB = i;
                        break;
                    }
                }
                int newSize = lastIndexOfA + sB.length - firstIndexOfB;
                int[] rsA = new int[newSize];
                int[] rsB = new int[newSize];
                for(int i=0; i<newSize; i++ ) {
                    if (i<=lastIndexOfA) rsA[ i ] = i; else rsA[ i ] = -1;
                    if (i>=lastIndexOfA) rsB[ i ] = i-lastIndexOfA+firstIndexOfB; else rsB[ i ] = -1;
                }
                return new int[][]{rsA, rsB};
            }

            @Contract(pure = true)
            public static int[] shpOfCon(int[] shp1, int[] shp2) {
                int[] shape = new int[(shp1.length + shp2.length) / 2];
                for ( int i = 0; i < shp1.length && i < shp2.length; i++) shape[ i ] = Math.abs(shp1[ i ] - shp2[ i ]) + 1;
                return shape;
            }

            @Contract(pure = true)
            public static int[] shpOfBrc(int[] shp1, int[] shp2) {
                int[] shape = new int[(shp1.length + shp2.length) / 2];
                for ( int i = 0; i < shp1.length && i < shp2.length; i++ ) {
                    shape[ i ] = Math.max(shp1[ i ], shp2[ i ]);
                    if (Math.min(shp1[ i ], shp2[ i ])!=1&&Math.max(shp1[ i ], shp2[ i ])!=shape[ i ]) {
                        throw new IllegalStateException("Broadcast not possible. Shapes do not match!");
                    }
                }
                return shape;
            }


        }

    }




}
