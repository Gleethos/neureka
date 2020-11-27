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
import neureka.ndim.config.NDConfiguration;
import neureka.utility.DataConverter;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;
import java.util.function.Consumer;


/**
 *  This is the precursor class to the final Tsr class from which
 *  tensor instances can be created.
 *  The inheritance model of a tensor is structured as follows:
 *  Tsr inherits from AbstractNDArray which inherits from AbstractComponentOwner
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 *
 */
public abstract class AbstractNDArray<InstanceType, ValueType> extends AbstractComponentOwner<InstanceType> implements Iterable<ValueType>
{

    /**
     *  An interface provided by sl4j which enables a modular logging backend!
     */
    protected static Logger _LOGGER; // Why is this not final ? : For unit testing!

    protected NDConfiguration _conf;

    private DataType<?> _dataType = DataType.instance( Neureka.instance().settings().dtype().getDefaultDataTypeClass() );
    
    private Object _data;

    public Class<?> getValueClass()
    {
        DataType<?> dt = _dataType;
        if ( dt != null ) return dt.getTypeClass();
        else return null;
    }

    public DataType getDataType() {
        return _dataType;
    }

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

    public Object getData() {
        return _data;
    }


    protected void _setData( Object data )
    {
        if ( _dataType == null ) {
            String message = "Trying to set data in a tensor which does not have a DataTyp instance.";
            _LOGGER.error( message );
            throw new IllegalStateException( message );
        }
        if ( data != null && _dataType.typeClassImplements( NumericType.class ) ) {
            NumericType numericType = (NumericType) _dataType.getTypeClassInstance();
            if ( numericType.targetArrayType() != data.getClass() ) {
                String message = "Cannot set data whose type does not match what is defined by the DataType instance.";
                _LOGGER.error( message );
                throw new IllegalStateException( message );
            }
        }
        _data = data;
    }

    protected void _allocate( int size )
    {
        _data = _dataType.allocate( size );
    }

    protected void _virtualize()
    {
        _data = _dataType.virtualize(_data);
    }

    protected void _actualize()
    {
        _data = _dataType.actualize(_data, this.size() );
    }

    protected Object _convertedDataOfType( Class<?> typeClass )
    {
        DataType newDT = DataType.instance( typeClass );
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
    public void forEach( Consumer<? super ValueType> action ) {
        for ( ValueType v : this ) action.accept( v );
    }

    @Override
    public Spliterator<ValueType> spliterator()
    {
        return new Spliterator<ValueType>()
        {
            @Override
            public boolean tryAdvance( Consumer<? super ValueType> action ) {
                return false;
            }

            @Override
            public Spliterator<ValueType> trySplit() {
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

    public abstract Object getValueAt( int i );

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public boolean is( Class<?> typeClass ) {
        DataType<?> type = DataType.instance( typeClass );
        return type == _dataType;
    }

    public boolean is64(){
        return _data instanceof double[];
    }

    public boolean is32(){
        return _data instanceof float[];
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public int i_of_i( int i ){
        return _conf.i_of_i( i );
    }

    public int[] idx_of_i( int i ) {
        return _conf.idx_of_i( i );
    }

    public int i_of_idx( int[] idx ) {
        return _conf.i_of_idx(idx);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public NDConfiguration getNDConf(){
        return _conf;
    }

    public InstanceType setNDConf( NDConfiguration ndConfiguration ){
        _conf = ndConfiguration;
        return (InstanceType) this;
    }


    //---

    public int rank(){
        return _conf.shape().length;
    }

    public List<Integer> shape() {
        return _asList(_conf.shape());
    }

    public int shape( int i ){
        return _conf.shape()[ i ];
    }

    public List<Integer> idxmap(){
        return _asList(_conf.idxmap());
    }

    public List<Integer> translation() {
        return _asList(_conf.translation());
    }

    public List<Integer> spread(){
        return _asList(_conf.spread());
    }

    public List<Integer> offset(){
        return _asList(_conf.offset());
    }

    public int size() {
        return NDConfiguration.Utility.szeOfShp(_conf.shape());
    }

    protected static List<Integer> _asList( int[] array ){
        List<Integer> intList = new ArrayList<>( array.length );
        for ( int i : array ) intList.add( i );
        return intList;
    }



    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    /**
     *  Static methods.
     */
    public static class Utility
    {
        public static class Stringify
        {
            @Contract( pure = true )
            public static String formatFP( double v ){
                DecimalFormatSymbols formatSymbols = new DecimalFormatSymbols( Locale.US );
                DecimalFormat Formatter = new DecimalFormat("##0.0##E0", formatSymbols);
                String vStr = String.valueOf( v );
                final int offset = 0;
                if ( vStr.length() > ( 7 - offset ) ){
                    if ( vStr.startsWith("0.") ) {
                        vStr = vStr.substring( 0, 7-offset )+"E0";
                    } else if( vStr.startsWith( "-0." ) ){
                        vStr = vStr.substring( 0, 8-offset )+"E0";
                    } else {
                        vStr = Formatter.format( v );
                        vStr = (!vStr.contains(".0E0"))?vStr:vStr.replace(".0E0",".0");
                        vStr = (vStr.contains("."))?vStr:vStr.replace("E0",".0");
                    }
                }
                return vStr;
            }

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
                                    " (" + Utility.Stringify.strConf(newShp) + ((NDConfiguration.Utility.szeOfShp(newShp) < t.size()) ? "<" : ">") + Utility.Stringify.strConf(t._conf.shape()) + ")");
                }
                return newShp;
            }

            @Contract(pure = true)
            public static int[][] makeFit( int[] sA, int[] sB ){
                int lastIndexOfA = 0;
                for ( int i = sA.length-1; i >= 0; i-- ) {
                    if(sA[ i ]!=1){
                        lastIndexOfA = i;
                        break;
                    }
                }
                int firstIndexOfB = 0;
                for (int i=0; i<sB.length; i++){
                    if(sB[ i ]!=1){
                        firstIndexOfB = i;
                        break;
                    }
                }
                int newSize = lastIndexOfA + sB.length - firstIndexOfB;
                int[] rsA = new int[newSize];
                int[] rsB = new int[newSize];
                for(int i=0; i<newSize; i++) {
                    if(i<=lastIndexOfA) rsA[ i ] = i; else rsA[ i ] = -1;
                    if(i>=lastIndexOfA) rsB[ i ] = i-lastIndexOfA+firstIndexOfB; else rsB[ i ] = -1;
                }
                return new int[][]{rsA, rsB};
            }

            @Contract(pure = true)
            public static int[] shpOfCon(int[] shp1, int[] shp2) {
                int[] shape = new int[(shp1.length + shp2.length) / 2];
                for (int i = 0; i < shp1.length && i < shp2.length; i++) shape[ i ] = Math.abs(shp1[ i ] - shp2[ i ]) + 1;
                return shape;
            }

            @Contract(pure = true)
            public static int[] shpOfBrc(int[] shp1, int[] shp2) {
                int[] shape = new int[(shp1.length + shp2.length) / 2];
                for (int i = 0; i < shp1.length && i < shp2.length; i++) {
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
