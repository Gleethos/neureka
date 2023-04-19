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

package neureka;

import neureka.autograd.GraphNode;
import neureka.common.composition.AbstractComponentOwner;
import neureka.common.utility.DataConverter;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.ndim.Filler;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;
import org.slf4j.Logger;

import java.util.Arrays;


/**
 *  This is the precursor class to the final {@link TsrImpl} class from which
 *  tensor instances can be created. <br>
 *  The inheritance model of a tensor is structured as follows: <br>
 *  {@link TsrImpl} inherits from {@link AbstractNda} which inherits from {@link AbstractComponentOwner}.
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 *  The above-mentioned classes are implementation details covered by
 *  the {@link Nda} and {@link Tsr} interfaces, which define various
 *  default methods spanning a rich API with good interoperability with
 *  different JVM languages...
 *
 * @param <C> The type of the concrete class extending this abstract class (currently the {@link Tsr} class).
 * @param <V> The value type of the individual items stored within this nd-array.
 */
abstract class AbstractNda<C, V> extends AbstractComponentOwner<Tsr<V>> implements Tsr<V>
{
    protected static Logger _LOG;

    /**
     *  An instance of an implementation of the {@link NDConfiguration} interface defining
     *  the dimensionality of this {@link AbstractNda} in terms of certain index properties
     *  which imply individual access patterns for the underlying {@link #_data}.
     */
    private NDConfiguration _NDConf;

    /**
     *  The heart and sole of the nd-array / tensor: its underlying data array.
     */
    private Data<V> _data;

    /**
     *  This integer represents the version of the data (accessible through {@link #getRawData()})
     *  stored within this tensor.
     *  It gets incremented every time an inline operation occurs!
     *  {@link GraphNode} instances tied to this tensor (as component) store
     *  a reference version which is a copy of this field.
     *  If this version changes, despite there being a GraphNode which might
     *  perform auto-differentiation at some point, then an exception will be thrown for debugging.
     *  <br>
     *  The corresponding getter returns the version of the data (accessible through {@link #getRawData()})
     *  stored within this tensor.
     */
    protected int _version = 0;


    protected final void _guardGet( String varName ) { _guard("Trying to access the "+varName+" of an already deleted tensor." ); }
    protected final void _guardSet( String varName ) { _guard("Trying to set the "+varName+" of an already deleted tensor." ); }
    protected final void _guardMod( String varName ) { _guard("Trying to modify the "+varName+" of an already deleted tensor." ); }

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

    /** {@inheritDoc} */
    @Override
    public NDConfiguration getNDConf() { _guardGet("ND-Configuration"); return _NDConf; }

    /** {@inheritDoc} */
     @Override
    public DataType<V> getDataType() {
         _guardGet("data type"); return _data != null && _data.dataType() != null ? _data.dataType() : null;
     }

    protected final Data<V> _getData() { _guardGet("data object"); return _data; }

    protected final Object _getRawData() {
        return  _getData() == null ? null : _getData().getOrNull();
    }

    /** {@inheritDoc} */
    @Override
    public Class<V> getItemType() {
        _guardGet("data type class"); return ( _data != null && _data.dataType() != null ? _data.dataType().getItemTypeClass() : null );
    }

    /** {@inheritDoc} */
     @Override
    public Class<?> getRepresentativeItemClass() {
        _guardGet("representative data type class"); return ( _data != null && _data.dataType() != null ? _data.dataType().getRepresentativeType() : null );
    }

    /**
     * @param array The data array managing the underlying data of this tensor/nd-array.
     *             This will be the same instance returned by {@link #_getData()}.
     */
    protected final void _setData( Data<V> array )
    {
        _guardSet( "data object" );
        Object data = array == null ? null : array.getOrNull();
        // Note: If the data is null, this might mean the tensor is outsourced (data is somewhere else)
        if ( _data != null && _data.getOrNull() != data && data != null && _data.getOrNull() != null ) {
            boolean isProbablyDeviceTransfer = ( _data.getOrNull().getClass().isArray() != data.getClass().isArray() );
            if ( !isProbablyDeviceTransfer)
                _version++; // Autograd must be warned!
        }
        _data = array;
    }

    protected <T> void _initDataArrayFrom( Filler<T> filler )
    {
        CPU.JVMExecutor executor = CPU.get().getExecutor();
        Object data = _getData().getOrNull();
        if ( data instanceof double[] )
            executor.threaded( ( (double[]) data ).length, ( start, end ) -> {
                for (int i = start; i < end; i++)
                    ( (double[]) data )[i] = ((Number) filler.init( i, _NDConf.indicesOfIndex(i))).doubleValue();
            });
        else if ( data instanceof float[] )
            executor.threaded( ( (float[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (float[]) data )[ i ] = ((Number) filler.init( i, _NDConf.indicesOfIndex( i ) )).floatValue();
            });
        else if ( data instanceof int[] )
            executor.threaded( ( (int[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (int[]) data )[ i ] = ((Number) filler.init( i, _NDConf.indicesOfIndex( i ) )).intValue();
            });
        else if ( data instanceof short[] )
            executor.threaded( ( (short[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (short[]) data )[ i ] = ((Number) filler.init( i, _NDConf.indicesOfIndex( i ) )).shortValue();
            });
        else if ( data instanceof byte[] )
            executor.threaded( ( (byte[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (byte[]) data )[ i ] = ((Number) filler.init( i, _NDConf.indicesOfIndex( i ) )).byteValue();
            });
        else if ( data instanceof long[] )
            executor.threaded( ( (long[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (long[]) data )[ i ] = ((Number) filler.init( i, _NDConf.indicesOfIndex( i ) )).longValue();
            });
        else if ( data instanceof boolean[] )
            executor.threaded( ( (boolean[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (boolean[]) data )[ i ] = (Boolean) filler.init( i, _NDConf.indicesOfIndex( i )  );
            });
        else if ( data instanceof char[] )
            executor.threaded( ( (char[]) data ).length, (start, end) -> {
                for (int i = start; i < end; i++)
                    ( (char[]) data )[ i ] = (Character) filler.init( i, _NDConf.indicesOfIndex( i )  );
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
     */
    protected final void _allocateVirtual() {
        _data = getDevice()
                .allocate(
                    this.getDataType(),
                    NDConstructor.of( this.getNDConf().shape() ).produceNDC(true)
                );
    }

    /**
     *  The internal implementation handling {@link MutateTsr#setIsVirtual(boolean)}.
     *
     * @param isVirtual The truth value determining if this should be made virtual or actual.
     */
    protected abstract void _setIsVirtual( boolean isVirtual );

    /**
     *  The {@link AbstractNda} is in essence a precursor class to the {@link Tsr} which encapsulates
     *  and protects most of its state...
     *  This is especially important during constructing where a wider range of unexpected user input
     *  might lead to a wider variety of exceptions.
     *  The API returned by this method simplifies this greatly.
     *
     * @return An {@link TsrConstructor} exposing a simple API for configuring a new {@link Tsr} instance.
     */
    protected static TsrConstructor constructFor( AbstractNda<?, ?> nda, Device<?> targetDevice, NDConstructor ndConstructor )
    {
        return
            new TsrConstructor(
                targetDevice, ndConstructor,
                new TsrConstructor.API() {
                    @Override public void setConf( NDConfiguration conf   ) { nda.getMut().setNDConf( conf ); }
                    @Override public void setData( Data o                 ) { nda._setData( o ); /*AbstractNda.this.set((Device)o.owner());*/ }
                    @Override public void setIsVirtual( boolean isVirtual ) { nda._setIsVirtual( isVirtual ); }
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
    protected final void _virtualize() { _data = getDevice().access(this).virtualize(); }

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
    protected final void _actualize() { _data = getDevice().access(this).actualize(); }

    protected Object _convertedDataOfType( Class<?> typeClass )
    {
        DataType<?> newDT = DataType.of( typeClass );
        if (
            newDT.typeClassImplements( NumericType.class )
                    &&
            getDataType().typeClassImplements( NumericType.class )
        ) {
            NumericType<?,Object, ?, Object> targetType  = (NumericType<?, Object,?, Object>) newDT.getTypeClassInstance(NumericType.class);
            return targetType.readForeignDataFrom( iterator(), this.size() );
        }
        else
            return DataConverter.get().convert( _getRawData(), newDT.getRepresentativeType() );
    }

    /**
     *  {@inheritDoc}
     */
     @Override
    public boolean is( Class<?> typeClass ) {
        DataType<?> type = DataType.of( typeClass );
        return type == _getData().dataType();
    }

    /**
     * This method sets the NDConfiguration of this NDArray.
     * Therefore, it should not be used lightly as it can cause major internal inconsistencies.
     *
     * @param ndConfiguration The new NDConfiguration instance which ought to be set.
     */
    protected void _setNDConf(NDConfiguration ndConfiguration )
    {
        _guardSet( "ND-Configuration" );
        if ( _NDConf != null && ndConfiguration != null ) {
            int s1 = Arrays.stream( _NDConf.shape() ).map( Math::abs ).reduce( 1, ( a, b ) -> a*b );
            int s2 = Arrays.stream( ndConfiguration.shape() ).map( Math::abs ).reduce( 1, ( a, b ) -> a*b );
            assert s1 == s2;
        }
        _NDConf = ndConfiguration;
        getDevice().access(this).updateNDConf();
    }

}
