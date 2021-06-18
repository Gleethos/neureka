package neureka.utility.fluent;

import groovy.lang.ObjectRange;
import neureka.Tsr;
import neureka.dtype.DataType;
import neureka.utility.fluent.states.IterByOrIterFromOrAll;
import neureka.utility.fluent.states.Step;
import neureka.utility.fluent.states.To;
import neureka.utility.fluent.states.WithShapeOrScalarOrVector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 *  This class exposes a fluent builder API for creating {@link Tsr} instances.
 *  An simple example would be:
 * <pre>{@code
 *
 *    Tsr.of(Double.class)
 *          .withShape( 2, 3, 4 )
 *          .iterativelyFilledBy( 5, 3, 5 )
 *
 * }</pre>
 *
 * It is also possible to define a range using the API to populate the tensor with values:
 * <pre>{@code
 *
 *    Tsr.of(Double.class)
 *          .withShape( 2, 3, 4 )
 *          .iterativelyFilledFrom( 2 ).to( 9 ).step( 2 )
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
 *
 *
 * @param <V> The type of the values which ought to be represent.
 */
public class TensorBuilder<V> implements WithShapeOrScalarOrVector<V>, IterByOrIterFromOrAll<V>, To<V>, Step<V>
{
    private final DataType<V> _dataType;
    private int[] _shape;
    private V _from;
    private V _to;

    public TensorBuilder( Class<V> typeClass ) { _dataType = DataType.of( typeClass ); }

    @SafeVarargs
    @Override
    public final Tsr<V> iterativelyFilledBy( V... values ) {
        return Tsr.of( _shape, _dataType, values );
    }

    @Override
    public To<V> iterativelyFilledFrom( V index ) {
        _from = index;
        return this;
    }

    @Override
    public Tsr<V> all( V value ) {
        return Tsr.of( _shape, _dataType, value );
    }

    @Override
    public IterByOrIterFromOrAll<V> withShape( int... shape ) {
        _shape = shape;
        return this;
    }

    @Override
    public Tsr<V> vector( Object[] values ) {
        return Tsr.of( new int[]{ values.length }, _dataType, values );
    }

    @Override
    public Tsr<V> scalar( V value ) {
        return Tsr.of( new int[]{1}, value.getClass(), value );
    }

    @Override
    public Step<V> to( V index ) {
        _to = index;
        return this;
    }


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
        else if ( _dataType == DataType.of( Float.class ) ) {
            List<Float> range = new ArrayList<>();
            for ( double index = ((Float) _from); index <= ((Float)_to) && itemIndex < itemLimit; index += size ) {
                range.add( (float) index );
                itemIndex++;
            }
            float[] primData = new float[tensorSize];
            for ( int ii = 0; ii < tensorSize; ii ++) {
                primData[ii] = range.get(ii%range.size());
            }
            data = primData;
        }
        else if ( _from instanceof Comparable && _to instanceof Comparable ) {
            data = new ObjectRange( (Comparable<V>) _from, (Comparable<V>) _to ).step( (int) size );
        }
        return Tsr.of( _shape, _dataType, data );
    }

    private int _size() {
        int size = 1;
        for ( int axis : _shape ) size *= axis;
        return size;
    }

}
