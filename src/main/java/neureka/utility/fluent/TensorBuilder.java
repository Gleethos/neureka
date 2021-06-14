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
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

    /*
        Tsr.forType(Double.class)
              .withShape( 2, 3, 4 )
              .iterativelyFilledBy( 5, 3, 5 )

        Tsr.forType(Double.class)
              .withShape( 2, 3, 4 )
              .iterativelyFilledFrom( 2 ).to( 9 ).step( 2 )

        Tsr.forType(Float.class).scalar( 3f )

        Tsr.forType(Byte.class).vector( 2, 5, 6, 7, 8 )

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
    public final Tsr<V> iterativelyFilledBy(V... values) {
        return new Tsr<>( _shape, _dataType, values );
    }

    @Override
    public To<V> iterativelyFilledFrom( V index ) {
        _from = index;
        return this;
    }

    @Override
    public Tsr<V> all( V value ) {
        return new Tsr<>( _shape, _dataType, value );
    }

    @Override
    public IterByOrIterFromOrAll<V> withShape( int... shape ) {
        _shape = shape;
        return this;
    }

    @Override
    public Tsr<V> vector( Object[] values ) {
        return new Tsr<>( new int[]{ values.length }, _dataType, values );
    }

    @Override
    public Tsr<V> scalar( V value ) {
        return new Tsr<>( value );
    }

    @Override
    public Step<V> to( V index ) {
        _to = index;
        return this;
    }


    @Override
    public Tsr<V> step( double size ) {
        Object data = null;
        if ( _dataType == DataType.of( Integer.class ) ) {
            List<Integer> range = new ArrayList<>();
            for ( int index = ((Integer) _from); index < ((Integer)_to); index += size )
                range.add( index );
            data = range.stream().mapToInt( v -> v ).toArray();
        }
        else if ( _dataType == DataType.of( Double.class ) ) {
            List<Double> range = new ArrayList<>();
            for ( double index = ((Double) _from); index < ((Double)_to); index += size )
                range.add( index );
            data = range.stream().mapToDouble( v -> v ).toArray();
        }
        else if ( _dataType == DataType.of( Float.class ) ) {
            List<Float> range = new ArrayList<>();
            for ( float index = ((Float) _from); index < ((Float)_to); index += size )
                range.add( index );
            float[] primData = new float[range.size()];
            for ( int i = 0; i < range.size(); i ++) {
                primData[i] = range.get(i);
            }
            data = primData;
        }
        else if ( _from instanceof Comparable && _to instanceof Comparable ) {
            data = new ObjectRange( (Comparable<V>) _from, (Comparable<V>) _to ).step( (int) size );
        }
        return new Tsr<V>( _shape, _dataType, data );
    }
}
