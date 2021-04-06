package neureka.utility;

import lombok.Setter;
import lombok.experimental.Accessors;
import neureka.Tsr;

public class SliceBuilder<ValType>
{
    private final Tsr _toBeSliced;
    private final AxisSliceBuilder<ValType>[]  _axisSliceBuilders;

    public SliceBuilder( Tsr toBeScliced ) {
        _toBeSliced = toBeScliced;
        int[] shape = toBeScliced.getNDConf().shape();
        _axisSliceBuilders = new AxisSliceBuilder[ shape.length ];
        for ( int i = 0; i < shape.length; i++ ) {
            _axisSliceBuilders[ i ] = new AxisSliceBuilder<>( shape[ i ], this );
        }

    }

    public AxisSliceBuilder sliceAxis( int axisIndex ) {
        if ( axisIndex >= _axisSliceBuilders.length ) throw new IllegalStateException("");
        return _axisSliceBuilders[ axisIndex ];
    }

    public Tsr<ValType> get() {
        return null;
    }

    @Accessors( fluent = true, prefix = {"_"}, chain = true )
    public static class AxisSliceBuilder<ValType>
    {
        private final SliceBuilder<ValType> _parent;
        private final int _axisSize;
        @Setter private int _from;
        @Setter private int _to;
        @Setter private int _steps;

        AxisSliceBuilder(int axisSize, SliceBuilder<ValType> parent) {
            _parent = parent;
            _axisSize = axisSize;
            _from = 0;
            _to = _axisSize - 1;
            _steps = 1;
        }

        public SliceBuilder<ValType> then() {
            return _parent;
        }
    }

}
