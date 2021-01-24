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

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Neureka;
import neureka.Tsr;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.ndim.config.NDConfiguration;
import neureka.utility.DataConverter;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Spliterator;
import java.util.function.Consumer;


/**
 *  This is the precursor class to the final Tsr class from which
 *  tensor instances can be created. <br>
 *  The inheritance model of a tensor is structured as follows: <br>
 *  Tsr inherits from AbstractNDArray which inherits from AbstractComponentOwner
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 */
@Accessors( prefix = {"_"} )
public abstract class AbstractNDArray<InstanceType, ValType> extends AbstractComponentOwner<InstanceType> implements Iterable<ValType>
{
    public interface Initializer<T> {  T init( int i, int[] index );  }

    /**
     *  An interface provided by sl4j which enables a modular logging backend!
     */
    protected static Logger _LOG; // Why is this not final ? : For unit testing!

    @Getter
    private NDConfiguration _NDConf;

    @Getter
    private DataType<?> _dataType = DataType.of( Neureka.instance().settings().dtype().getDefaultDataTypeClass() );

    @Getter
    private Object _data;

    public Class<?> getValueClass()
    {
        DataType<?> dt = _dataType;
        if ( dt != null ) return dt.getTypeClass();
        else return null;
    }

    /**
     *  This method enables modifying the data-type configuration of this NDArray.
     *  Warning! The method should not be used unless absolutely necessary.
     *  This is because it can cause unpredictable inconsistencies between the
     *  underlying DataType instance of this NDArray and the actual type of the actual
     *  data it is wrapping (or it is referencing on a Device).
     *
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
            NumericType numericType = (NumericType) _dataType.getTypeClassInstance();
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
            for ( int i=0; i<((double[])data).length; i++ )
                ( (double[]) data )[i] = (double) initializer.init( i, _NDConf.idx_of_i( i )  );
        else if ( data instanceof float[] )
            for ( int i=0; i<((float[])data).length; i++ )
                ( (float[]) data )[i] = (float) initializer.init( i, _NDConf.idx_of_i( i )  );
        else if ( data instanceof int[] )
            for ( int i=0; i<((int[])data).length; i++ )
                ( (int[]) data )[i] = (int) initializer.init( i, _NDConf.idx_of_i( i )  );
        else if ( data instanceof short[] )
            for ( int i=0; i<((short[])data).length; i++ )
                ( (short[]) data )[i] = (short) initializer.init( i, _NDConf.idx_of_i( i )  );
        else if ( data instanceof byte[] )
            for ( int i=0; i<((byte[])data).length; i++ )
                ( (byte[]) data )[i] = (byte) initializer.init( i, _NDConf.idx_of_i( i )  );
        else
            for ( int i=0; i<((Object[])data).length; i++ )
                ( (Object[]) data )[i] = initializer.init( i, _NDConf.idx_of_i( i )  );

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
    protected void _allocate( int size )
    {
        _data = _dataType.allocate( size );
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

    public int i_of_i( int i ) {
        return _NDConf.i_of_i( i );
    }

    public int[] idx_of_i( int i ) {
        return _NDConf.idx_of_i( i );
    }

    public int i_of_idx( int[] idx ) {
        return _NDConf.i_of_idx(idx);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  This method sets the NDConfiguration of this NDArray.
     *  Therefore, it should not be used lightly as it can cause major internal inconsistencies.
     *
     * @param ndConfiguration The new NDConfiguration instance which ought to be set.
     * @return The final instance type of this class which enables method chaining.
     */
    public InstanceType setNDConf( NDConfiguration ndConfiguration ) {
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

    public List<Integer> idxmap() {
        return _asList(_NDConf.idxmap());
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
            public static int[] shpCheck( int[] newShp, Tsr t ) {
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
                return newShp;
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
