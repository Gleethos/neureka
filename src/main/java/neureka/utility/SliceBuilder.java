package neureka.utility;

import lombok.Setter;
import lombok.experimental.Accessors;
import neureka.Tsr;

import java.util.function.Supplier;


public class SliceBuilder<ValType>
{
    interface CreationCallback<V> { Tsr<V> sliceOf(int[] newShape, int[] newOffset, int[] newSpread ); }

    private final Supplier<Tsr<ValType>> _create;
    private final AxisSliceBuilder<ValType>[]  _axisSliceBuilders;

    public SliceBuilder( Tsr<ValType> toBeSliced, CreationCallback<ValType> sliceCreator ) {
        int[] shape = toBeSliced.getNDConf().shape();
        _axisSliceBuilders = new AxisSliceBuilder[ shape.length ];
        int[] newShape = new int[shape.length];
        int[] newSpread = new int[shape.length];
        int[] newOffset = new int[shape.length];
        for ( int i = 0; i < shape.length; i++ ) {
            int finalI = i;
            _axisSliceBuilders[ i ] = new AxisSliceBuilder<>(
                                                shape[ i ],
                                                (from, to, step) -> {
                                                    newOffset[ finalI ] = from;
                                                    newShape[ finalI ] = to;
                                                    newSpread[ finalI ] = step;
                                                    return this;
                                                } );
        }
        _create = () -> sliceCreator.sliceOf( newShape, newOffset, newSpread );
    }

    public AxisSliceBuilder<ValType> sliceAxis( int axisIndex ) {
        if ( axisIndex >= _axisSliceBuilders.length ) throw new IllegalStateException("");
        return _axisSliceBuilders[ axisIndex ];
    }

    public Tsr<ValType> get() {
        return _create.get();
    }

    @Accessors( fluent = true, prefix = {"_"}, chain = true )
    public static class AxisSliceBuilder<ValType>
    {
        interface Resolution<V> { SliceBuilder<V> resolve( int from, int to, int steps ); }

        private final Resolution<ValType> _then;
        private final int _axisSize;
        @Setter private int _from;
        @Setter private int _to;
        @Setter private int _steps = 1;

        AxisSliceBuilder( int axisSize, Resolution<ValType> then ) {
            _then = then;
            _axisSize = axisSize;
            _from = 0;
            _to = _axisSize - 1;
        }

        public SliceBuilder<ValType> then() {
            while ( _to < 0 ) _to += _axisSize;
            while ( _from < 0 ) _from += _axisSize;
            _to %= _axisSize;
            _from %= _axisSize;
            return _then.resolve(_from, _to, _steps);
        }
    }

}
