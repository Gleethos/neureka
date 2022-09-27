package neureka.fluent.building;

import neureka.Nda;
import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.fluent.building.states.*;
import neureka.ndim.Filler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

/**
 *  This is the implementation of the fluent builder API for creating {@link Nda}/{@link Tsr} instances.
 *  A simple example would be:
 * <pre>{@code
 *
 *    Tsr.of(Double.class)
 *          .withShape( 2, 3, 4 )
 *          .andFill( 5, 3, 5 )
 *
 * }</pre>
 *
 * It is also possible to define a range using the API to populate the tensor with values:
 * <pre>{@code
 *
 *    Tsr.of(Double.class)
 *          .withShape( 2, 3, 4 )
 *          .andFillFrom( 2 ).to( 9 ).step( 2 )
 *
 * }</pre>
 *
 * If one needs a simple scalar then the following shortcut is possible:
 * <pre>{@code
 *
 *    Tsr.of(Float.class).scalar( 3f )
 *
 * }</pre>
 *
 * This principle works for vectors as well:
 * <pre>{@code
 *
 *     Tsr.of(Byte.class).vector( 2, 5, 6, 7, 8 )
 *
 * }</pre>
 * For more fine-grained control over the initialization one can
 * pass an initialization lambda to the API:
 * <pre>{@code
 *
 *     Tsr.of(Byte.class).withShape(2, 3).andWhere( (i, indices) -> i * 5 - 30 )
 *
 * }</pre>
 *
 * @param <V> The type of the values which ought to be represented by the {@link Tsr} built by this {@link NdaBuilder}.
 */
public final class NdaBuilder<V> implements WithShapeOrScalarOrVectorOnDevice<V>, IterByOrIterFromOrAllTsr<V>, ToForTsr<V>, StepForTsr<V>
{
    private static final Logger _LOG = LoggerFactory.getLogger(NdaBuilder.class);

    private final DataType<V> _dataType;
    private int[] _shape;
    private V _from;
    private V _to;
    private Device<V> _device = (Device<V>) CPU.get();

    /**
     * @param typeClass The type of the values which ought to be represented by the {@link Tsr} built by this {@link NdaBuilder}.
     */
    public NdaBuilder( Class<V> typeClass ) {
        LogUtil.nullArgCheck( typeClass, "typeClass", Class.class, "Cannot build tensor without data type information!" );
        _dataType = DataType.of( typeClass );
    }

    private Tsr<V> _get( Object value ) {
        LogUtil.nullArgCheck( value, "value", Object.class, "Cannot build tensor where value is null!" );
        return Tsr.of( _dataType, _shape, value ).to( _device );
    }

    /**
     * @param values The values which will recurrently populate the returned {@link Tsr} with values until it is filled.
     * @return A new {@link Tsr} instance populated by the array of values supplied to this method.
     */
    @SafeVarargs
    @Override
    public final Tsr<V> andFill( V... values ) {
        LogUtil.nullArgCheck( values, "values", _dataType.getItemTypeClass(), "Cannot fill a tensor will a value array that is null!" );
        if ( _isAllOne(values) ) return _get( values[0] );
        return _get( values );
    }

    private <T> boolean _isAllOne(T[] values) {
        if ( values.length > 0 ) {
            T first = values[0];
            if ( values.length == 1 ) return true;
            else if ( values.length <= 42 ) {
                for ( T value : values ) {
                    if ( !Objects.equals(first, value) )
                        return false;
                }
                return true;
            }
        }
        return false;
    }

    /**
     *  This method receives an {@link Filler} lambda which will be
     *  used to populate the {@link Tsr} instance produced by this API with values.
     *
     * @param filler The {@link Filler} which ought to populate the returned {@link Tsr}.
     * @return A new {@link Tsr} instance populated by the lambda supplied to this method.
     */
    @Override
    public Tsr<V> andWhere( Filler<V> filler ) {
        return Tsr.of( _dataType, _shape, filler ).to( _device );
    }

    @Override
    public ToForTsr<V> andFillFrom( V index ) {
        LogUtil.nullArgCheck(index, "index", _dataType.getItemTypeClass(), "Cannot create a range where the last index is undefined!");
        _from = _checked(index);
        return this;
    }

    @Override
    public Tsr<V> all( V value ) { return _get( value ); }

    @Override
    public Tsr<V> andSeed( Object seed ) {
        Class<V> type = _dataType.getItemTypeClass();
        Class<?> seedType = seed.getClass();
        try {
            Function random = Neureka.get().backend().getFunction().random();
            if (type == Double.class && seedType == Long.class)
                return random.with( Arg.Seed.of((Long) seed) ).call( _get( 0d ) );
            else if (type == Float.class && seedType == Long.class)
                return random.with( Arg.Seed.of((Long) seed) ).call( _get( 0f ) );
            else
                return Tsr.of( type, _shape, seed.toString() ).to( _device );
        } catch ( Exception e ) {
            IllegalArgumentException exception =
                    new IllegalArgumentException(
                         "Could not create a random tensor for type '"+type+"'!"
                    );
            _LOG.error( exception.getMessage(), e );
            throw exception;
        }
    }

    @Override
    public IterByOrIterFromOrAllTsr<V> withShape( int... shape ) {
        LogUtil.nullArgCheck(shape, "shape", int[].class, "Cannot create a tensor without shape!");
        if ( shape.length == 0 )
            throw new IllegalArgumentException("Cannot instantiate a tensor without shape arguments.");
        _shape = shape;
        return this;
    }

    @Override
    public Tsr<V> vector( Object[] values ) {
        LogUtil.nullArgCheck(values, "values", Object[].class, "Cannot create a vector without data array!");
        _shape = new int[]{ values.length };
        if ( _isAllOne(values) ) return _get( values[0] );
        return _get( values );
    }

    @Override
    public Tsr<V> scalar( V value ) {
        if ( value != null ) {
            value = _checked( value );
            if ( value.getClass() != _dataType.getItemTypeClass() )
                throw new IllegalArgumentException("Provided value is of the wrong type!");
        }
        _shape = new int[]{ 1 };
        return _get( value );
    }

    /**
     *  This method makes sure that the data provided by the user is indeed of the right type
     *  by converting it if possible to the previously provided data type.
     *
     * @param o The scalar value which may need to be converted to the provided data type.
     * @return The value converted to the type defined by the provided {@link #_dataType}.
     */
    private V _checked( V o ) {
        Class<?> jvmType = _dataType.getItemTypeClass();
        if ( Number.class.isAssignableFrom(jvmType) ) {
            if ( o instanceof Number && o.getClass() != jvmType ) {
                Number n = (Number) o;
                if ( jvmType == Integer.class ) return (V) ((Integer) n.intValue()   );
                if ( jvmType == Double.class  ) return (V) ((Double)  n.doubleValue());
                if ( jvmType == Short.class   ) return (V) ((Short)   n.shortValue() );
                if ( jvmType == Byte.class    ) return (V) ((Byte)    n.byteValue()  );
                if ( jvmType == Long.class    ) return (V) ((Long)    n.longValue()  );
                if ( jvmType == Float.class   ) return (V) ((Float)   n.floatValue() );
            }
        }
        return o;
    }

    @Override
    public StepForTsr<V> to( V index ) { _to = _checked(index); return this; }

    @Override
    public Tsr<V> step( double size ) {
        int tensorSize = _size();
        Object data = null;
        int itemLimit = _size();
        int itemIndex = 0;
        if ( _dataType == DataType.of( Integer.class ) ) {
            List<Integer> range = new ArrayList<>();
            for ( int index = ((Integer) _from); index <= ((Integer)_to) && itemIndex < itemLimit; index += size ) {
                range.add( index );
                itemIndex++;
            }
            data = IntStream.iterate( 0, i -> i + 1 )
                            .limit( tensorSize )
                            .map( i -> range.get( i % range.size() ) )
                            .toArray();
        }
        else if ( _dataType == DataType.of( Double.class ) ) {
            List<Double> range = new ArrayList<>();
            for ( double index = ((Double) _from); index <= ((Double)_to) && itemIndex < itemLimit; index += size ) {
                range.add( index );
                itemIndex++;
            }
            data = IntStream.iterate( 0, i -> i + 1 )
                            .limit( tensorSize )
                            .mapToDouble( i -> range.get( i % range.size() ) )
                            .toArray();
        }
        else if ( _dataType == DataType.of( Long.class ) ) {
            List<Long> range = new ArrayList<>();
            for ( long index = ((Long) _from); index <= ((Long)_to) && itemIndex < itemLimit; index += size ) {
                range.add( index );
                itemIndex++;
            }
            data = IntStream.iterate( 0, i -> i + 1 )
                    .limit( tensorSize )
                    .mapToLong( i -> range.get( i % range.size() ) )
                    .toArray();
        }
        else if ( _dataType == DataType.of( Float.class ) ) {
            List<Float> range = new ArrayList<>();
            for ( double index = ((Float) _from); index <= ((Float)_to) && itemIndex < itemLimit; index += size ) {
                range.add( (float) index );
                itemIndex++;
            }
            float[] primData = new float[ tensorSize ];
            for ( int ii = 0; ii < tensorSize; ii++ )
                primData[ ii ] = range.get( ii % range.size() );

            data = primData;
        }
        else if ( _dataType == DataType.of( Byte.class ) ) {
            List<Byte> range = new ArrayList<>();
            for ( byte index = ((Byte) _from); index <= ((Byte)_to) && itemIndex < itemLimit; index += size ) {
                range.add( index );
                itemIndex++;
            }
            byte[] primData = new byte[ tensorSize ];
            for ( int ii = 0; ii < tensorSize; ii++ )
                primData[ ii ] = range.get( ii % range.size() );

            data = primData;
        }
        else if ( _from instanceof Comparable && _to instanceof Comparable ) {
            //data = new ObjectRange( (Comparable<V>) _from, (Comparable<V>) _to ).step( (int) size );
            throw new IllegalStateException("Cannot form a range for the provided elements...");
            // TODO: make it possible to have ranges like 'a' to 'z'...
        }
        return _get( data );
    }

    private int _size() {
        int size = 1;
        for ( int axis : _shape ) size *= axis;
        return size;
    }

    @Override
    public WithShapeOrScalarOrVectorTsr<V> on( Device<V> device ) {
        LogUtil.nullArgCheck(device, "device", Device.class, "Cannot create a tensor with an undefined device!");
        _device = device;
        return this;
    }
}
