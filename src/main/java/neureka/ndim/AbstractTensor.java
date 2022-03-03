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
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.common.composition.AbstractComponentOwner;
import neureka.common.utility.DataConverter;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;

import java.util.Arrays;


/**
 *  This is the precursor class to the final {@link Tsr} class from which
 *  tensor instances can be created. <br>
 *  The inheritance model of a tensor is structured as follows: <br>
 *  {@link Tsr} inherits from {@link AbstractTensor} which inherits from {@link AbstractComponentOwner}
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 *  This class also implements the {@link TensorAPI} interface, which uses
 *  default methods to expose a rich API with good interoperability with
 *  different JVM languages...
 *
 * @param <C> The type of the concrete class extending this abstract class (currently the {@link Tsr} class).
 * @param <V> The value type of the individual items stored within this nd-array.
 */
public abstract class AbstractTensor<C, V> extends AbstractComponentOwner<C> implements TensorAPI<V>
{
    /**
     *  An interface provided by sl4j which enables a modular logging backend!
     */
    protected static Logger _LOG;

    /**
     *  An instance of an implementation of the {@link NDConfiguration} interface defining
     *  the dimensionality of this {@link AbstractTensor} in terms of certain index properties
     *  which imply individual access patterns for the underlying {@link #_data}.
     */
    private NDConfiguration _NDConf;
    /**
     *  The data type instance wrapping and representing the actual value type class.
     */
    private DataType<?> _dataType = DataType.of( Neureka.get().settings().dtype().getDefaultDataTypeClass() );
    /**
     *  The heart and sole of the nd-array / tensor: its underlying data array.
     */
    private Object _data;

    /**
     * @return The truth value determining if the {@link Unsafe#delete()} method has been called oin this instance.
     */
    public abstract boolean isDeleted();

    protected void _guardGet( String varName ) { _guard("Trying to access the "+varName+" of an already deleted tensor." ); }
    protected void _guardSet( String varName ) { _guard("Trying to set the "+varName+" of an already deleted tensor." ); }
    protected void _guardMod( String varName ) { _guard("Trying to modify the "+varName+" of an already deleted tensor." ); }

    /**
     *  This method will guard the state of deleted tensors by throwing an {@link IllegalAccessError}
     *  if this {@link Tsr} has already been deleted and whose state should no longer be exposed to
     *  anything but the garbage collector...
     *
     * @param message The message explaining to the outside which kind of access violation just occurred.
     */
    private void _guard( String message ) {
        if ( this.isDeleted() ) {
            _LOG.error( message );
            throw new IllegalAccessError( message );
        }
    }

    /**
     * @return The {@link NDConfiguration} implementation instance of this {@link Tsr} storing dimensionality information.
     */
    @Override
    public NDConfiguration getNDConf() { _guardGet("ND-Configuration"); return _NDConf; }

    /**
     *  This method returns the {@link DataType} instance of this {@link Tsr}, which is
     *  a wrapper object for the actual type class representing the value items stored inside
     *  the underlying data array of this tensor.
     *
     * @return The {@link DataType} instance of this {@link Tsr} storing important type information.
     */
    public DataType<?> getDataType() { _guardGet("data type"); return _dataType; }

    /**
     *  This returns the underlying raw data object of this tensor.
     *  Contrary to the {@link Tsr#getValue()} ()} method, this one will
     *  return an unbiased view on the data of this tensor.
     *
     * @return The raw data object underlying this tensor.
     */
    public Object getData() { _guardGet("data object"); return _data; }

    /**
     * @return The type class of individual value items within this {@link Tsr} instance.
     */
    @Override
    public Class<V> getValueClass() {
        _guardGet("data type class"); return ( _dataType != null ? (Class<V>) _dataType.getJVMTypeClass() : null );
    }

    /**
     *  The {@link Class} returned by this method is the representative {@link Class} of the
     *  value items of a concrete {@link AbstractTensor} but not necessarily the actual {@link Class} of
     *  a given value item, this is especially true for numeric types, which are represented by
     *  implementations of the {@link NumericType} interface.                                        <br>
     *  For example in the case of a tensor of type {@link Double}, this method would
     *  return {@link neureka.dtype.custom.F64} which is the representative class of {@link Double}. <br>
     *  Calling the {@link #getValueClass()} method instead of this method would return the actual value
     *  type class, namely: {@link Double}.
     *
     * @return The representative type class of individual value items within this concrete {@link AbstractTensor}
     *         extension instance which might also be sub-classes of the {@link NumericType} interface
     *         to model unsigned types or other JVM foreign numeric concepts.
     */
    public Class<?> getRepresentativeValueClass() {
        _guardGet("representative data type class"); return ( _dataType != null ? _dataType.getTypeClass() : null );
    }

    /**
     *  This method enables modifying the data-type configuration of this {@link AbstractTensor}.
     *  Warning! The method should not be used unless absolutely necessary.
     *  This is because it can cause unpredictable inconsistencies between the
     *  underlying {@link DataType} instance of this {@link AbstractTensor} and the actual type of the actual
     *  data it is wrapping (or it is referencing on a {@link neureka.devices.Device}).<br>
     *  <br>
     * @param dataType The new {@link DataType} which ought to be set.
     * @return The final instance type of this class which enables method chaining.
     */
    protected C _setDataType( DataType<?> dataType )
    {
        _guardSet( "data type" );
        if ( _data != null ) {
            String message = "Data type of tensor can only be set when data attribute is null!\n" +
                             "This is due to construction-consistency reasons.\n";
            throw new IllegalStateException( message );
        }
        _dataType = dataType;
        return (C) this;
    }

    /**
     * @param data The data object which ought to be set for this array.
     *             This will be the same instance returned by {@link #getData()}.
     */
    protected void _setData( Object data )
    {
        _guardSet( "data object" );
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

    protected <T> void _initData( Filler<T> filler)
    {
        CPU.JVMExecutor executor = CPU.get().getExecutor();
        Object data = getData();
        if ( data instanceof double[] )
            executor.threaded( ( (double[]) data ).length, ( start, end ) -> {
                for (int i = start; i < end; i++)
                    ( (double[]) data )[i] = (double) filler.init(i, _NDConf.indicesOfIndex(i));
            });
        else if ( data instanceof float[] )
            executor.threaded( ( (float[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (float[]) data )[ i ] = (float) filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
        else if ( data instanceof int[] )
            executor.threaded( ( (int[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (int[]) data )[ i ] = (int) filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
        else if ( data instanceof short[] )
            executor.threaded( ( (short[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (short[]) data )[ i ] = (short) filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
        else if ( data instanceof byte[] )
            executor.threaded( ( (byte[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (byte[]) data )[ i ] = (byte) filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
        else if ( data instanceof long[] )
            executor.threaded( ( (long[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (long[]) data )[ i ] = (long) filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
        else if ( data instanceof boolean[] )
            executor.threaded( ( (boolean[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (boolean[]) data )[ i ] = (boolean) filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
        else if ( data instanceof char[] )
            executor.threaded( ( (char[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (char[]) data )[ i ] = (char) filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
        else
            executor.threaded( ( (Object[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (Object[]) data )[ i ] = filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
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

    /**
     *  WARNING! Virtualizing is the process of compacting the underlying data array
     *  down to an array holding a single value.
     *  This only makes sense for homogeneously populated tensors.
     *  Passing {@code false} to this method will "actualize" a "virtual" tensor.
     *  Meaning the underlying data array will at least become as large as the size of the tensor
     *  as is defined by {@link #size()}.
     *
     * @param isVirtual The truth value determining if this tensor should be "virtual" or "actual".
     * @return This concrete instance, to allow for method chaining.
     */
    public abstract C setIsVirtual( boolean isVirtual );

    /**
     *  A Virtual tensor is a tensor whose underlying data array is of size 1, holding only a single value. <br>
     *  This only makes sense for homogeneously populated tensors.
     *  An example of such a tensor would be: <br>
     *  {@code Tsr.ofInts().withShape(x,y).all(n)}                           <br><br>
     *
     *  Use {@link #setIsVirtual(boolean)} to "actualize" a "virtual" tensor, and vise versa.
     *
     * @return The truth value determining if this tensor is "virtual" or "actual".
     */
    public abstract boolean isVirtual();

    /**
     *  The internal implementation handling {@link #setIsVirtual(boolean)}.
     *
     * @param isVirtual The truth value determining if this should be made virtual or actual.
     */
    protected abstract void _setIsVirtual( boolean isVirtual );

    /**
     *  The {@link AbstractTensor} is in essence a precursor class to the {@link Tsr} which encapsulates
     *  and protects most of its state...
     *  This is especially important during constructing where a wider range of unexpected user input
     *  might lead to a wider variety of exceptions.
     *  The API returned by this method simplifies this greatly.
     *
     * @return An {@link TsrConstructor} exposing a simple API for configuring a new {@link Tsr} instance.
     */
    protected TsrConstructor createConstructionAPI()
    {
        AbstractTensor<C, ?> nda = this;
        return new TsrConstructor(
                    new TsrConstructor.API() {
                        @Override public void setType( DataType<?> type        ) { nda.getUnsafe().setDataType( type ); }
                        @Override public void setConf( NDConfiguration conf    ) { nda.getUnsafe().setNDConf(   conf ); }
                        @Override public void setData( Object o                ) { nda._setData(      o  ); }
                        @Override public void allocate( int size               ) { nda._allocate(   size ); }
                        @Override public Object getData()                        { return nda.getData();    }
                        @Override public void setIsVirtual( boolean isVirtual )  { nda._setIsVirtual( isVirtual ); }
                    }
                );
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
    protected void _virtualize() { _data = _dataType.virtualize( _data ); }

    /**
     *  An actual NDArray (tensor) is the opposite to a virtual one. <br>
     *  Virtual means that the size of the underlying data does not match the real size of the NDArray.
     *  This is the case when the NDArray is filled with one element homogeneously.
     *  An example would be an all zeros array. The reasoning behind this feature is memory efficiency.
     *  It would be unreasonable to allocate an array filled entirely with one and the same value item!<br>
     *  <br>
     *  This method turns the data of a virtual NDArray into a newly allocated data array matching the
     *  size of the nd-array type... <br>
     */
    protected void _actualize() { _data = _dataType.actualize( _data, this.size() ); }

    protected Object _convertedDataOfType( Class<?> typeClass )
    {
        DataType<?> newDT = DataType.of( typeClass );
        if (
            newDT.typeClassImplements( NumericType.class )
                    &&
            getDataType().typeClassImplements( NumericType.class )
        ) {
            NumericType<?,Object, ?, Object> targetType  = (NumericType<?, Object,?, Object>) newDT.getTypeClassInstance();
            return targetType.readForeignDataFrom( iterator(), this.size() );
        }
        else
            return DataConverter.instance().convert( getData(), newDT.getTypeClass() );
    }

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

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  This method sets the NDConfiguration of this NDArray.
     *  Therefore, it should not be used lightly as it can cause major internal inconsistencies.
     *
     * @param ndConfiguration The new NDConfiguration instance which ought to be set.
     * @return The final instance type of this class which enables method chaining.
     */
    protected C _setNDConf( NDConfiguration ndConfiguration )
    {
        _guardSet( "ND-Configuration" );
        if ( _NDConf != null && ndConfiguration != null ) {
            int s1 = Arrays.stream( _NDConf.shape() ).map( Math::abs ).reduce( 1, ( a, b ) -> a*b );
            int s2 = Arrays.stream( ndConfiguration.shape() ).map( Math::abs ).reduce( 1, ( a, b ) -> a*b );
            assert s1 == s2;
        }
        _NDConf = ndConfiguration;
        if ( this.has( Device.class ) ) {
            this.get(Device.class).updateNDConf((Tsr) this);
        }
        return (C) this;
    }

    /**
     *  This method exposes an API for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptional situations!<br>
     *  <br><b>
     *
     *  Only use this if you know what you are doing and
     *  performance is critical! <br>
     *  </b>
     *  (Like custom backend extensions for example)
     *
     * @return The unsafe API exposes methods for mutating the state of the tensor.
     */
    public abstract Unsafe<V> getUnsafe();

    /**
     *  Tensors should be considered immutable, however sometimes it
     *  is important to mutate their state for performance reasons.
     *  This interface exposes several methods for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptions...<br>
     *  <br>
     */
    public interface Unsafe<T> {
        /**
         *  This method sets the NDConfiguration of this NDArray.
         *  Therefore, it should not be used lightly as it can cause major internal inconsistencies.
         *
         * @param configuration The new NDConfiguration instance which ought to be set.
         * @return The final instance type of this class which enables method chaining.
         */
        Tsr<T> setNDConf( NDConfiguration configuration );
        /**
         *  This method is an inline operation which changes the underlying data of this tensor.
         *  It converts the data types of the elements of this tensor to the specified type!<br>
         *  <br>
         *  <b>WARNING : The usage of this method is discouraged for the following reasons: </b><br>
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
         * @param <V> The type parameter for the returned tensor.
         * @return The same tensor instance whose data has been converted to hold a different type.
         */
        <V> Tsr<V> toType( Class<V> typeClass );

        /**
         *  This method enables modifying the data-type configuration of this {@link AbstractTensor}.
         *  Warning! The method should not be used unless absolutely necessary.
         *  This is because it can cause unpredictable inconsistencies between the
         *  underlying {@link DataType} instance of this {@link AbstractTensor} and the actual type of the actual
         *  data it is wrapping (or it is referencing on a {@link neureka.devices.Device}).<br>
         *  <br>
         * @param dataType The new {@link DataType} which ought to be set.
         * @return The tensor with the new data type set.
         */
        <V> Tsr<V> setDataType( DataType<V> dataType );

        /**
         *  This method allows you to modify the data-layout of this {@link AbstractTensor}.
         *  Warning! The method should not be used unless absolutely necessary.
         *  This is because it can cause unpredictable side effects especially for certain
         *  operations expecting a particular data layout (like for example matrix multiplication).
         *  <br>
         *
         * @param layout The layout of the data array (row or column major).
         * @return The final instance type of this class which enables method chaining.
         */
        Tsr<T> toLayout( NDConfiguration.Layout layout );

        /**
         *  This method is responsible for incrementing
         *  the "_version" field variable which represents the version of the data of this tensor.
         *  Meaning :
         *  Every time the underlying data (_value) changes this version ought to increment alongside.
         *  The method is called during the execution procedure.
         *
         * @param call The context object containing all relevant information that defines a call for tensor execution.
         * @return This very tensor instance. (factory pattern)
         */
        Tsr<T> incrementVersion( ExecutionCall<?> call );

        /**
         *  Intermediate tensors are internal non-user tensors which may be eligible
         *  for deletion when further consumed by a {@link Function}.
         *  For the casual user of Neureka, this flag should always be false!
         *
         * @param isIntermediate The truth value determining if this tensor is not a user tensor but an internal
         *                       tensor which may be eligible for deletion by {@link Function}s consuming it.
         */
        Tsr<T> setIsIntermediate( boolean isIntermediate );


        /**
         *  Although tensors will be garbage collected when they are not strongly referenced,
         *  there is also the option to manually free up the tensor and its associated data in a native environment.
         *  This is especially useful when tensors are stored on a device like the {@link neureka.devices.opencl.OpenCLDevice}.
         *  In that case calling this method will free the memory reserved for this tensor on the device.
         *  This manual memory freeing through this method can be faster than waiting for
         *  the garbage collector to kick in at a latr point in time... <br>
         *  <br>
         *
         * @return This very tensor instance to allow for method chaining.
         */
        Tsr<T> delete();

    }

    /**
     *  Static utility methods for the NDArray.
     */
    public static class Utility
    {
        @Contract( pure = true )
        public static String shapeString( int[] conf ) {
            StringBuilder str = new StringBuilder();
            for ( int i = 0; i < conf.length; i++ )
                str.append(conf[ i ]).append((i != conf.length - 1) ? ", " : "");
            return "[" + str + "]";
        }

        @Contract(pure = true)
        public static int[][] makeFit( int[] sA, int[] sB ) {
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

}
